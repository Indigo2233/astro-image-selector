//! CPU star detection using a DAOFIND-style kernel and background estimator.
//! See `licenses/PHOTUTILS-AND-SCIPY.txt` for the algorithm notices.

use std::collections::HashMap;
use std::sync::LazyLock;

const KERNEL_RADIUS: usize = 2;
const KERNEL_SIZE: usize = 2 * KERNEL_RADIUS + 1;
const KERNEL_PIXELS: usize = KERNEL_SIZE * KERNEL_SIZE;
const DAO_FWHM: f64 = 3.0;
const PIXEL_WEIGHTS: [f64; KERNEL_SIZE] = [1.0, 2.0, 3.0, 2.0, 1.0];

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Star {
    pub x: f64,
    pub y: f64,
    pub peak: f64,
    pub flux: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DetectionConfig {
    pub center_region_ratio: f64,
    pub threshold: f64,
    pub edge_exclusion: usize,
}

impl Default for DetectionConfig {
    fn default() -> Self {
        Self {
            center_region_ratio: 0.5,
            threshold: 5.0,
            edge_exclusion: 50,
        }
    }
}

#[derive(Default)]
pub struct StarDetector {
    backgrounds: HashMap<(usize, usize, u64), BackgroundStats>,
}

#[derive(Clone, Copy)]
struct BackgroundStats {
    threshold: f64,
    level: f64,
}

impl StarDetector {
    pub fn new() -> Self {
        Self::default()
    }

    /// The optimized path reuses the first frame's background for a given
    /// crop size and threshold during this detector's lifetime.
    pub fn detect(
        &mut self,
        pixels: &[f64],
        width: usize,
        height: usize,
        config: &DetectionConfig,
    ) -> Vec<Star> {
        self.detect_cancellable(pixels, width, height, config, || false)
    }

    pub fn detect_cancellable(
        &mut self,
        pixels: &[f64],
        width: usize,
        height: usize,
        config: &DetectionConfig,
        cancelled: impl Fn() -> bool,
    ) -> Vec<Star> {
        if cancelled() {
            return Vec::new();
        }
        let Some(mut crop) = Crop::new(pixels, width, height, config.center_region_ratio) else {
            return Vec::new();
        };
        let key = (crop.width, crop.height, config.threshold.to_bits());
        let stats = if let Some(stats) = self.backgrounds.get(&key) {
            *stats
        } else {
            let Some(background) = BackgroundMaps::new(&crop) else {
                // Both source paths use the same background estimator.
                return Vec::new();
            };
            let stats = BackgroundStats {
                threshold: background.threshold(config.threshold),
                level: median(&crop.pixels),
            };
            self.backgrounds.insert(key, stats);
            stats
        };
        for value in &mut crop.pixels {
            *value -= stats.level;
        }
        crop.find_stars(
            stats.threshold,
            width,
            height,
            config.edge_exclusion,
            &cancelled,
        )
    }

    /// The original uncached path subtracts the interpolated background map
    /// from the crop instead of subtracting one scalar background level.
    pub fn detect_unoptimized(
        &self,
        pixels: &[f64],
        width: usize,
        height: usize,
        config: &DetectionConfig,
    ) -> Vec<Star> {
        let Some(mut crop) = Crop::new(pixels, width, height, config.center_region_ratio) else {
            return Vec::new();
        };
        let Some(background) = BackgroundMaps::new(&crop) else {
            return Vec::new();
        };
        let threshold = background.threshold(config.threshold);
        for (value, level) in crop.pixels.iter_mut().zip(background.level) {
            *value -= level;
        }
        crop.find_stars(threshold, width, height, config.edge_exclusion, &|| false)
    }
}

struct Crop {
    pixels: Vec<f64>,
    width: usize,
    height: usize,
    left: usize,
    top: usize,
}

impl Crop {
    fn new(pixels: &[f64], width: usize, height: usize, ratio: f64) -> Option<Self> {
        if width.checked_mul(height)? != pixels.len() || !ratio.is_finite() || ratio <= 0.0 {
            return None;
        }
        let bounds = |size: usize| {
            let center = size / 2;
            let half = (size as f64 * ratio / 2.0) as usize;
            (
                center.saturating_sub(half),
                center.saturating_add(half).min(size),
            )
        };
        let (left, right) = bounds(width);
        let (top, bottom) = bounds(height);
        if right - left < KERNEL_SIZE || bottom - top < KERNEL_SIZE {
            return None;
        }
        let pixels = (top..bottom)
            .flat_map(|y| pixels[y * width + left..y * width + right].iter().copied())
            .collect();
        Some(Self {
            pixels,
            width: right - left,
            height: bottom - top,
            left,
            top,
        })
    }

    fn find_stars(
        &self,
        threshold: f64,
        image_width: usize,
        image_height: usize,
        edge: usize,
        cancelled: &dyn Fn() -> bool,
    ) -> Vec<Star> {
        if self.pixels.iter().all(|pixel| !pixel.is_finite()) || threshold.is_nan() {
            return Vec::new();
        }
        let kernel = &*DAO_KERNEL;
        let threshold = threshold * kernel.relative_error;
        let convolved = self.convolve(kernel);
        if convolved.iter().all(|&pixel| pixel == convolved[0]) {
            return Vec::new();
        }
        let min = convolved.iter().copied().fold(f64::INFINITY, f64::min);
        let for_peak_search = |value: f64| if value.is_nan() { min } else { value };
        let x_max = image_width as f64 - edge as f64;
        let y_max = image_height as f64 - edge as f64;
        let mut stars = Vec::new();
        for y in KERNEL_RADIUS..self.height - KERNEL_RADIUS {
            if cancelled() {
                return Vec::new();
            }
            for x in KERNEL_RADIUS..self.width - KERNEL_RADIUS {
                let peak = for_peak_search(convolved[y * self.width + x]);
                if peak <= threshold {
                    continue;
                }
                let conv_window = cutout(&convolved, self.width, x, y);
                if kernel
                    .footprint
                    .iter()
                    .any(|&index| for_peak_search(conv_window[index]) > peak)
                {
                    continue;
                }
                let window = cutout(&self.pixels, self.width, x, y);
                if let Some(mut star) = kernel.measure(&window, &conv_window, threshold) {
                    star.x += (self.left + x) as f64;
                    star.y += (self.top + y) as f64;
                    if star.x > edge as f64
                        && star.x < x_max
                        && star.y > edge as f64
                        && star.y < y_max
                    {
                        stars.push(star);
                    }
                }
            }
        }
        stars
    }

    fn convolve(&self, kernel: &DaoKernel) -> Vec<f64> {
        let mut convolved = vec![0.0; self.pixels.len()];
        for y in 0..self.height {
            for x in 0..self.width {
                let mut value = 0.0;
                for &index in &kernel.footprint {
                    let row = y as isize + (index / KERNEL_SIZE) as isize - KERNEL_RADIUS as isize;
                    let col = x as isize + (index % KERNEL_SIZE) as isize - KERNEL_RADIUS as isize;
                    if (0..self.height as isize).contains(&row)
                        && (0..self.width as isize).contains(&col)
                    {
                        value += self.pixels[row as usize * self.width + col as usize]
                            * kernel.weights[index];
                    }
                }
                convolved[y * self.width + x] = value;
            }
        }
        convolved
    }
}

fn cutout(data: &[f64], width: usize, x: usize, y: usize) -> [f64; KERNEL_PIXELS] {
    let mut window = [0.0; KERNEL_PIXELS];
    for (row, target) in window
        .as_chunks_mut::<KERNEL_SIZE>()
        .0
        .iter_mut()
        .enumerate()
    {
        let start = (y + row - KERNEL_RADIUS) * width + x - KERNEL_RADIUS;
        target.copy_from_slice(&data[start..start + KERNEL_SIZE]);
    }
    window
}

static DAO_KERNEL: LazyLock<DaoKernel> = LazyLock::new(DaoKernel::new);

struct DaoKernel {
    footprint: Vec<usize>,
    weights: [f64; KERNEL_PIXELS],
    relative_error: f64,
    profile: MarginalProfile,
}

impl DaoKernel {
    fn new() -> Self {
        let sigma = DAO_FWHM / (8.0 * 2.0_f64.ln()).sqrt();
        let sigma_squared = sigma.powi(2);
        let mut footprint = Vec::new();
        let mut gaussian = [0.0; KERNEL_PIXELS];
        let mut masked_sum = 0.0;
        let mut masked_squared_sum = 0.0;
        for (index, sample) in gaussian.iter_mut().enumerate() {
            let x = index % KERNEL_SIZE;
            let y = index / KERNEL_SIZE;
            let radius_squared =
                x.abs_diff(KERNEL_RADIUS).pow(2) + y.abs_diff(KERNEL_RADIUS).pow(2);
            *sample = (-(radius_squared as f64) / (2.0 * sigma_squared)).exp();
            if radius_squared <= KERNEL_RADIUS.pow(2) {
                footprint.push(index);
                masked_sum += *sample;
                masked_squared_sum += sample.powi(2);
            }
        }
        let mean = masked_sum / footprint.len() as f64;
        let variance_sum = masked_squared_sum - masked_sum * mean;
        let mut weights = [0.0; KERNEL_PIXELS];
        for &index in &footprint {
            weights[index] = (gaussian[index] - mean) / variance_sum;
        }
        Self {
            footprint,
            weights,
            relative_error: 1.0 / variance_sum.sqrt(),
            profile: MarginalProfile::new(&gaussian, sigma_squared),
        }
    }

    fn measure(
        &self,
        data: &[f64; KERNEL_PIXELS],
        convolved: &[f64; KERNEL_PIXELS],
        threshold: f64,
    ) -> Option<Star> {
        let center = KERNEL_PIXELS / 2;
        let peak = data[center];
        let peak_neighbor_mean = (self.footprint.iter().map(|&i| data[i]).sum::<f64>() - peak)
            / (self.footprint.len() - 1) as f64;
        let sharpness = (peak - peak_neighbor_mean) / convolved[center];
        if !(0.2..=1.0).contains(&sharpness) {
            return None;
        }
        let roundness1 = quadrants_roundness(convolved);
        if !(-1.0..=1.0).contains(&roundness1) {
            return None;
        }
        let (x, hx) = self.profile.fit(data, true)?;
        let (y, hy) = self.profile.fit(data, false)?;
        let roundness2 = 2.0 * (hx - hy) / (hx + hy);
        let flux = data.iter().sum::<f64>();
        if !(-1.0..=1.0).contains(&roundness2)
            || !peak.is_finite()
            || (!flux.is_finite() && threshold != 0.0)
        {
            return None;
        }
        Some(Star { x, y, peak, flux })
    }
}

fn quadrants_roundness(convolved: &[f64; KERNEL_PIXELS]) -> f64 {
    let mut quadrants = [0.0; 4];
    let mut sum_abs = 0.0;
    for (index, &value) in convolved.iter().enumerate() {
        let x = index % KERNEL_SIZE;
        let y = index / KERNEL_SIZE;
        if x == KERNEL_RADIUS && y == KERNEL_RADIUS {
            continue;
        }
        let quadrant = if y <= KERNEL_RADIUS && x > KERNEL_RADIUS {
            0
        } else if y < KERNEL_RADIUS && x <= KERNEL_RADIUS {
            1
        } else if y >= KERNEL_RADIUS && x < KERNEL_RADIUS {
            2
        } else {
            3
        };
        quadrants[quadrant] += value;
        sum_abs += value.abs();
    }
    2.0 * (-quadrants[0] + quadrants[1] - quadrants[2] + quadrants[3]) / sum_abs
}

struct MarginalProfile {
    kernel: [f64; KERNEL_SIZE],
    derivative: [f64; KERNEL_SIZE],
    weighted_sum: f64,
    height_denominator: f64,
    derivative_sum: f64,
    kernel_derivative_sum: f64,
    derivative_energy: f64,
    sigma_squared: f64,
}

impl MarginalProfile {
    const WEIGHT_SUM: f64 = 9.0;

    fn new(gaussian: &[f64; KERNEL_PIXELS], sigma_squared: f64) -> Self {
        let kernel = weighted_profile(gaussian, true);
        let derivative = std::array::from_fn(|i| kernel[i] * (KERNEL_RADIUS as f64 - i as f64));
        let squared_sum = weighted_sum(|i| kernel[i].powi(2));
        let kernel_sum = weighted_sum(|i| kernel[i]);
        Self {
            kernel,
            derivative,
            weighted_sum: kernel_sum,
            height_denominator: squared_sum - kernel_sum.powi(2) / Self::WEIGHT_SUM,
            derivative_sum: weighted_sum(|i| derivative[i]),
            kernel_derivative_sum: weighted_sum_product(&kernel, &derivative),
            derivative_energy: weighted_sum(|i| derivative[i].powi(2)),
            sigma_squared,
        }
    }

    fn fit(&self, data: &[f64; KERNEL_PIXELS], along_x: bool) -> Option<(f64, f64)> {
        let profile = weighted_profile(data, along_x);
        let sum = weighted_sum(|i| profile[i]);
        let data_kernel = weighted_sum_product(&profile, &self.kernel);
        let numerator = data_kernel - (sum * self.weighted_sum) / Self::WEIGHT_SUM;
        if numerator <= 0.0 || self.height_denominator <= 0.0 {
            return None;
        }
        let height = numerator / self.height_denominator;
        let data_derivative = weighted_sum_product(&profile, &self.derivative);
        let mut shift = (self.kernel_derivative_sum
            - (data_derivative - self.derivative_sum * sum))
            / (height * self.derivative_energy / self.sigma_squared);
        let half_size = KERNEL_SIZE as f64 / 2.0;
        if shift.abs() > half_size {
            let moment = weighted_sum(|i| profile[i] * (KERNEL_RADIUS as f64 - i as f64));
            let direction = if along_x { 1.0 } else { -1.0 };
            shift = if sum == 0.0 {
                0.0
            } else {
                direction * moment / sum
            };
            if shift.abs() > half_size {
                shift = 0.0;
            }
        }
        (shift.is_finite() && height.is_finite()).then_some((shift, height))
    }
}

fn weighted_profile(data: &[f64; KERNEL_PIXELS], along_x: bool) -> [f64; KERNEL_SIZE] {
    std::array::from_fn(|position| {
        weighted_sum(|sample| {
            let index = if along_x {
                sample * KERNEL_SIZE + position
            } else {
                position * KERNEL_SIZE + sample
            };
            data[index]
        })
    })
}

fn weighted_sum(value: impl Fn(usize) -> f64) -> f64 {
    (0..KERNEL_SIZE).map(|i| value(i) * PIXEL_WEIGHTS[i]).sum()
}

fn weighted_sum_product(left: &[f64; KERNEL_SIZE], right: &[f64; KERNEL_SIZE]) -> f64 {
    weighted_sum(|i| left[i] * right[i])
}

struct BackgroundMaps {
    level: Vec<f64>,
    rms_median: f64,
}

impl BackgroundMaps {
    fn new(crop: &Crop) -> Option<Self> {
        let box_size = (crop.width.min(crop.height) / 6).clamp(30, 150);
        let box_width = box_size.min(crop.width);
        let box_height = box_size.min(crop.height);
        let grid_width = crop.width.div_ceil(box_width);
        let grid_height = crop.height.div_ceil(box_height);
        let mut levels = Vec::with_capacity(grid_width * grid_height);
        let mut rms = Vec::with_capacity(levels.capacity());
        let mut samples = Vec::with_capacity(box_width * box_height);
        let min_samples = 0.9 * (box_width * box_height) as f64;
        for grid_y in 0..grid_height {
            for grid_x in 0..grid_width {
                samples.clear();
                let left = grid_x * box_width;
                let right = (left + box_width).min(crop.width);
                let top = grid_y * box_height;
                let bottom = (top + box_height).min(crop.height);
                for y in top..bottom {
                    samples.extend(
                        crop.pixels[y * crop.width + left..y * crop.width + right]
                            .iter()
                            .copied()
                            .filter(|sample| sample.is_finite()),
                    );
                }
                let (level, deviation) =
                    clipped_statistics(&mut samples, min_samples).unwrap_or((f64::NAN, f64::NAN));
                levels.push(level);
                rms.push(deviation);
            }
        }
        let mesh_shape = (grid_width, grid_height);
        let output_shape = (crop.width, crop.height);
        let box_shape = (box_width, box_height);
        let level = background_map(levels, mesh_shape, output_shape, box_shape)?;
        let mut rms = background_map(rms, mesh_shape, output_shape, box_shape)?;
        Some(Self {
            rms_median: median_mut(&mut rms),
            level,
        })
    }

    fn threshold(&self, sigma_multiplier: f64) -> f64 {
        sigma_multiplier * self.rms_median
    }
}

fn clipped_statistics(samples: &mut Vec<f64>, min_samples: f64) -> Option<(f64, f64)> {
    for _ in 0..10 {
        if samples.len() as f64 <= min_samples {
            return None;
        }
        let center = median(samples);
        let deviation = population_std(samples);
        let low = center - 3.0 * deviation;
        let high = center + 3.0 * deviation;
        let previous_length = samples.len();
        samples.retain(|&value| !(value < low || value > high));
        if samples.len() == previous_length {
            return Some((center, deviation));
        }
    }
    (samples.len() as f64 > min_samples).then(|| (median(samples), population_std(samples)))
}

fn population_std(samples: &[f64]) -> f64 {
    let mean = samples.iter().sum::<f64>() / samples.len() as f64;
    let sum_squares = samples
        .iter()
        .map(|sample| (sample - mean).powi(2))
        .sum::<f64>();
    (sum_squares / samples.len() as f64).sqrt()
}

fn median(data: &[f64]) -> f64 {
    if data.is_empty() || data.iter().any(|value| value.is_nan()) {
        return f64::NAN;
    }
    median_mut(&mut data.to_vec())
}

fn median_mut(data: &mut [f64]) -> f64 {
    let length = data.len();
    let (lower_half, middle, _) = data.select_nth_unstable_by(length / 2, f64::total_cmp);
    if length.is_multiple_of(2) {
        let lower = lower_half.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        return (lower + *middle) / 2.0;
    }
    *middle
}

fn background_map(
    mesh: Vec<f64>,
    mesh_shape: (usize, usize),
    output_shape: (usize, usize),
    box_shape: (usize, usize),
) -> Option<Vec<f64>> {
    let (width, height) = mesh_shape;
    let mesh = fill_mesh(mesh, width)?;
    let filtered = median_filter_mesh(&mesh, width, height);
    Some(zoom_mesh(filtered, mesh_shape, output_shape, box_shape))
}

fn fill_mesh(mut mesh: Vec<f64>, width: usize) -> Option<Vec<f64>> {
    if mesh.iter().all(|sample| !sample.is_nan()) {
        return Some(mesh);
    }
    let known: Vec<_> = mesh
        .iter()
        .enumerate()
        .filter_map(|(index, &sample)| (!sample.is_nan()).then_some((index, sample)))
        .collect();
    if known.is_empty() {
        return None;
    }
    let mut neighbors = Vec::with_capacity(known.len());
    for (index, unknown) in mesh.iter_mut().enumerate().filter(|(_, v)| v.is_nan()) {
        neighbors.clear();
        neighbors.extend(known.iter().map(|&(other_index, sample)| {
            let dx = (index % width).abs_diff(other_index % width) as f64;
            let dy = (index / width).abs_diff(other_index / width) as f64;
            (dx * dx + dy * dy, other_index, sample)
        }));
        // Use row-major order for equally distant cells at the tenth-neighbor cutoff.
        let compare = |left: &(f64, usize, f64), right: &(f64, usize, f64)| {
            left.0
                .total_cmp(&right.0)
                .then_with(|| left.1.cmp(&right.1))
        };
        let count = neighbors.len().min(10);
        if count < neighbors.len() {
            neighbors.select_nth_unstable_by(count, compare);
        }
        neighbors[..count].sort_unstable_by(compare);
        let mut weighted_sum = 0.0;
        let mut weight_sum = 0.0;
        for &(squared_distance, _, sample) in &neighbors[..count] {
            let weight = 1.0 / squared_distance.sqrt();
            weighted_sum += weight * sample;
            weight_sum += weight;
        }
        *unknown = weighted_sum / weight_sum;
    }
    Some(mesh)
}

fn median_filter_mesh(mesh: &[f64], width: usize, height: usize) -> Vec<f64> {
    let mut filtered = Vec::with_capacity(mesh.len());
    let mut window = Vec::with_capacity(25);
    for y in 0..height {
        for x in 0..width {
            window.clear();
            let left = x.saturating_sub(2);
            let right = (x + 3).min(width);
            for row in y.saturating_sub(2)..(y + 3).min(height) {
                window.extend_from_slice(&mesh[row * width + left..row * width + right]);
            }
            filtered.push(median_mut(&mut window));
        }
    }
    filtered
}

fn zoom_mesh(
    mut coefficients: Vec<f64>,
    mesh_shape: (usize, usize),
    output_shape: (usize, usize),
    box_shape: (usize, usize),
) -> Vec<f64> {
    let (mesh_width, mesh_height) = mesh_shape;
    let (output_width, output_height) = output_shape;
    let min = coefficients.iter().copied().fold(f64::INFINITY, f64::min);
    let max = coefficients
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    if min == max {
        return vec![min; output_width * output_height];
    }
    prefilter_mesh(&mut coefficients, mesh_width, mesh_height);
    let x_samples = spline_axis(output_width, mesh_width, box_shape.0);
    let y_samples = spline_axis(output_height, mesh_height, box_shape.1);
    let mut output = Vec::with_capacity(output_width * output_height);
    for y in &y_samples {
        for x in &x_samples {
            let mut sample = 0.0;
            for (&row, &wy) in y.indices.iter().zip(&y.weights) {
                for (&col, &wx) in x.indices.iter().zip(&x.weights) {
                    sample += coefficients[row * mesh_width + col] * wy * wx;
                }
            }
            output.push(sample.clamp(min, max));
        }
    }
    output
}

fn prefilter_mesh(coefficients: &mut [f64], width: usize, height: usize) {
    let mut line = vec![0.0; height];
    for x in 0..width {
        for (row, value) in coefficients.chunks_exact(width).zip(&mut line) {
            *value = row[x];
        }
        prefilter_line(&mut line);
        for (row, &value) in coefficients.chunks_exact_mut(width).zip(&line) {
            row[x] = value;
        }
    }
    for row in coefficients.chunks_exact_mut(width) {
        prefilter_line(row);
    }
}

fn prefilter_line(line: &mut [f64]) {
    if line.len() <= 1 {
        return;
    }
    const POLE: f64 = -0.267_949_192_431_122_7;
    let gain = (1.0 - POLE) * (1.0 - 1.0 / POLE);
    for coefficient in &mut *line {
        *coefficient *= gain;
    }
    let end = line.len() - 1;
    let pole_n = POLE.powi(line.len() as i32);
    let mut pole_i = POLE;
    let first = line[0];
    line[0] += pole_n * line[end];
    for i in 1..=end {
        line[0] += pole_i * (line[i] + pole_n * line[end - i]);
        pole_i *= POLE;
    }
    line[0] *= POLE / (1.0 - pole_n.powi(2));
    line[0] += first;
    for i in 1..=end {
        line[i] += POLE * line[i - 1];
    }
    line[end] *= POLE / (POLE - 1.0);
    for i in (0..end).rev() {
        line[i] = POLE * (line[i + 1] - line[i]);
    }
}

struct SplineSample {
    indices: [usize; 4],
    weights: [f64; 4],
}

fn spline_axis(length: usize, mesh_length: usize, box_size: usize) -> Vec<SplineSample> {
    let zoom = 1.0 / box_size as f64;
    (0..length)
        .map(|index| {
            let coordinate = (index as f64 + 0.5) * zoom - 0.5;
            let coordinate = if mesh_length == 1 { 0.0 } else { coordinate };
            let base = coordinate.floor();
            let first = base as isize - 1;
            let fraction = coordinate - base;
            let complement = 1.0 - fraction;
            let inner_weight = |x: f64| (x.powi(2) * (x - 2.0) * 3.0 + 4.0) / 6.0;
            let outer_left = complement * complement * complement / 6.0;
            let inner_left = inner_weight(fraction);
            let inner_right = inner_weight(complement);
            SplineSample {
                indices: std::array::from_fn(|i| reflect(first + i as isize, mesh_length)),
                weights: [
                    outer_left,
                    inner_left,
                    inner_right,
                    1.0 - outer_left - inner_left - inner_right,
                ],
            }
        })
        .collect()
}

fn reflect(index: isize, length: usize) -> usize {
    let index = index.rem_euclid(2 * length as isize) as usize;
    if index < length {
        index
    } else {
        2 * length - index - 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn measures_subpixel_centroid_and_flux() {
        let image: Vec<_> = (0..64)
            .flat_map(|y| {
                (0..64).map(move |x| {
                    let dx = (f64::from(x) - 30.25) / 1.25;
                    let dy = (f64::from(y) - 32.75) / 1.35;
                    395.0
                        + f64::from((19 * x + 17 * y) % 11)
                        + 2400.0 * (-0.5 * (dx.powi(2) + dy.powi(2))).exp()
                })
            })
            .collect();
        let config = DetectionConfig {
            center_region_ratio: 1.0,
            edge_exclusion: 3,
            ..DetectionConfig::default()
        };
        let mut detector = StarDetector::new();
        let stars = detector.detect(&image, 64, 64, &config);
        assert_eq!(stars.len(), 1);
        let star = stars[0];
        assert!((star.x - 30.256_326_653_979_404).abs() < 1e-9);
        assert!((star.y - 32.743_836_338_871_81).abs() < 1e-9);
        assert!((star.peak - 2_316.483_281_942_092_6).abs() < 1e-8);
        assert!((star.flux - 22_805.455_353_184_55).abs() < 1e-8);
    }

    #[test]
    fn detects_star_after_removing_large_background() {
        let width = 96;
        let height = 96;
        let background = 10_000.0;
        let image: Vec<_> = (0..height)
            .flat_map(|y| {
                (0..width).map(move |x| {
                    let dx = (x as f64 - 48.35) / 1.35;
                    let dy = (y as f64 - 47.65) / 1.25;
                    background
                        + ((13 * x + 17 * y) % 7) as f64
                        + 90.0 * (-0.5 * (dx.powi(2) + dy.powi(2))).exp()
                })
            })
            .collect();
        let config = DetectionConfig {
            center_region_ratio: 1.0,
            threshold: 5.0,
            edge_exclusion: 5,
        };
        let mut detector = StarDetector::new();
        let stars = detector.detect(&image, width, height, &config);
        assert_eq!(stars.len(), 1);
        assert!((stars[0].x - 48.35).abs() < 0.2);
        assert!((stars[0].y - 47.65).abs() < 0.2);
    }

    #[test]
    fn returns_no_stars_for_flat_and_invalid_images() {
        let mut detector = StarDetector::new();
        let config = DetectionConfig::default();
        let flat = vec![1000.0; 150 * 150];
        assert!(detector.detect(&flat, 150, 150, &config).is_empty());
        assert!(detector.detect(&flat, 150, 153, &config).is_empty());
        assert!(
            detector
                .detect(
                    &flat,
                    150,
                    150,
                    &DetectionConfig {
                        center_region_ratio: 0.0,
                        ..config
                    }
                )
                .is_empty()
        );
    }
}
