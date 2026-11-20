use std::cell::Cell;
use std::cmp::Ordering;

use levenberg_marquardt::{LeastSquaresProblem, LevenbergMarquardt};
use nalgebra::{DVector, Dyn, Matrix, SVector, U7, storage::Owned};

const MAX_FIT_EVALUATIONS: usize = 10_000;
const DIFFERENCE_STEP: f64 = 1.490_116_119_384_765_6e-8;
const PARAMETERS: usize = 7;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Candidate {
    pub x: f64,
    pub y: f64,
    pub peak: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct MeasurementConfig {
    pub saturation_level: f64,
    pub max_stars: usize,
    pub fit_radius: usize,
}

impl Default for MeasurementConfig {
    fn default() -> Self {
        Self {
            saturation_level: 65_000.0,
            max_stars: 50,
            fit_radius: 15,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Measurement {
    pub fwhm: f64,
    pub eccentricity: f64,
    pub used_stars: usize,
}

/// Selects unsaturated stars. At the limit, later input rows win ties in peak value.
pub fn select_candidates(stars: &[Candidate], config: &MeasurementConfig) -> Vec<Candidate> {
    let mut selected: Vec<_> = stars
        .iter()
        .copied()
        .filter(|star| star.peak < config.saturation_level)
        .collect();
    if selected.len() > config.max_stars {
        selected.sort_by(|a, b| a.peak.partial_cmp(&b.peak).unwrap_or(Ordering::Equal));
        selected.reverse();
        selected.truncate(config.max_stars);
    }
    selected
}

pub fn measure(
    pixels: &[f64],
    width: usize,
    height: usize,
    stars: &[Candidate],
    config: &MeasurementConfig,
) -> Measurement {
    measure_cancellable(pixels, width, height, stars, config, || false).unwrap_or_default()
}

/// Returns None when the caller cancels the measurement.
pub fn measure_cancellable(
    pixels: &[f64],
    width: usize,
    height: usize,
    stars: &[Candidate],
    config: &MeasurementConfig,
    cancelled: impl Fn() -> bool,
) -> Option<Measurement> {
    if cancelled() {
        return None;
    }
    let radius = config.fit_radius;
    let Some(diameter) = radius.checked_mul(2).and_then(|size| size.checked_add(1)) else {
        return Some(Measurement::default());
    };
    if stars.is_empty()
        || diameter > width
        || diameter > height
        || width.checked_mul(height) != Some(pixels.len())
    {
        return Some(Measurement::default());
    }

    let mut cutout = Vec::with_capacity(diameter * diameter);
    let mut fwhm_values = Vec::new();
    let mut eccentricity_values = Vec::new();
    for star in select_candidates(stars, config) {
        if cancelled() {
            return None;
        }
        let (x, y) = (star.x.round_ties_even(), star.y.round_ties_even());
        if !(x >= radius as f64
            && x < (width - radius) as f64
            && y >= radius as f64
            && y < (height - radius) as f64)
        {
            continue;
        }

        cutout.clear();
        let (x, y) = (x as usize, y as usize);
        for row in y - radius..=y + radius {
            let offset = row * width + x - radius;
            cutout.extend_from_slice(&pixels[offset..offset + diameter]);
        }
        if let Some((fwhm, eccentricity)) =
            fit_star_cancellable(&cutout, diameter, diameter, &cancelled)
        {
            fwhm_values.push(fwhm);
            eccentricity_values.push(eccentricity);
        }
    }
    if cancelled() {
        return None;
    }
    let used_stars = fwhm_values.len();
    let (fwhm, _) = median_and_mad(&mut fwhm_values);
    let (eccentricity, _) = median_and_mad(&mut eccentricity_values);
    Some(Measurement {
        fwhm,
        eccentricity,
        used_stars,
    })
}

/// Returns the median and the mean absolute deviation from the median.
pub fn median_and_mad(values: &mut [f64]) -> (f64, f64) {
    if values.is_empty() {
        return (0.0, 0.0);
    }
    let median = median_in_place(values);
    let deviation = values
        .iter()
        .map(|value| (value - median).abs())
        .sum::<f64>()
        / values.len() as f64;
    (median, deviation)
}

pub fn snr(pixels: &[f64]) -> f64 {
    if pixels.is_empty() {
        return 0.0;
    }
    let mut scratch = pixels.to_vec();
    let median = median_in_place(&mut scratch);
    for (deviation, pixel) in scratch.iter_mut().zip(pixels) {
        *deviation = (pixel - median).abs();
    }
    let deviation = 1.4826 * median_in_place(&mut scratch);
    if deviation.is_nan() || deviation <= 0.0 {
        return 0.0;
    }

    let threshold = median + 3.0 * deviation;
    let (signal, count) = pixels
        .iter()
        .copied()
        .filter(|pixel| *pixel > threshold)
        .fold((0.0, 0), |(sum, count), pixel| (sum + pixel, count + 1));
    if count == 0 {
        return 0.0;
    }
    (signal / count as f64 - median) / deviation
}

fn median_in_place(values: &mut [f64]) -> f64 {
    if values.is_empty() || values.iter().any(|value| value.is_nan()) {
        return f64::NAN;
    }
    let len = values.len();
    let (lower, middle, _) = values.select_nth_unstable_by(len / 2, f64::total_cmp);
    if len.is_multiple_of(2) {
        let lower_middle = lower.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        (lower_middle + *middle) / 2.0
    } else {
        *middle
    }
}

pub fn fit_star(pixels: &[f64], width: usize, height: usize) -> Option<(f64, f64)> {
    fit_star_cancellable(pixels, width, height, &|| false)
}

fn fit_star_cancellable(
    pixels: &[f64],
    width: usize,
    height: usize,
    cancelled: &dyn Fn() -> bool,
) -> Option<(f64, f64)> {
    if width.checked_mul(height) != Some(pixels.len()) || pixels.len() < PARAMETERS {
        return None;
    }
    let mut min = pixels[0];
    let mut max = pixels[0];
    let mut peak = 0;
    for (index, &pixel) in pixels.iter().enumerate() {
        if !pixel.is_finite() {
            return None;
        }
        min = min.min(pixel);
        if pixel > max {
            max = pixel;
            peak = index;
        }
    }
    let initial = [
        max - min,
        (peak % width) as f64,
        (peak / width) as f64,
        3.0,
        3.0,
        0.0,
        min,
    ];
    let problem = GaussianFit {
        pixels,
        width,
        parameters: SVector::from(initial),
        evaluations: Cell::new(0),
        cancelled,
    };
    let (problem, report) = LevenbergMarquardt::new()
        .with_patience(MAX_FIT_EVALUATIONS / (PARAMETERS + 1))
        .minimize(problem);
    if !report.termination.was_successful() {
        return None;
    }
    let [_, x, y, sigma_x, sigma_y, _, _] = problem.parameters.into();
    let (sigma_x, sigma_y) = (sigma_x.abs(), sigma_y.abs());
    let fwhm = 2.3548 * (sigma_x * sigma_y).sqrt();
    let ratio = sigma_x.min(sigma_y) / sigma_x.max(sigma_y);
    let eccentricity = (1.0 - ratio * ratio).sqrt();
    let (width, height) = (width as f64, height as f64);
    if width / 4.0 < x
        && x < width * 3.0 / 4.0
        && height / 4.0 < y
        && y < height * 3.0 / 4.0
        && 0.5 < fwhm
        && fwhm < width / 2.0
        && (0.0..=1.0).contains(&eccentricity)
    {
        Some((fwhm, eccentricity))
    } else {
        None
    }
}

type FitParameters = SVector<f64, PARAMETERS>;
type Jacobian = Matrix<f64, Dyn, U7, Owned<f64, Dyn, U7>>;

struct GaussianFit<'a> {
    pixels: &'a [f64],
    width: usize,
    parameters: FitParameters,
    evaluations: Cell<usize>,
    cancelled: &'a dyn Fn() -> bool,
}

impl GaussianFit<'_> {
    fn residual_values(&self, parameters: &FitParameters) -> impl Iterator<Item = f64> + '_ {
        let [
            amplitude,
            center_x,
            center_y,
            sigma_x,
            sigma_y,
            theta,
            offset,
        ] = <[f64; PARAMETERS]>::from(*parameters);
        let cos = theta.cos();
        let sin = theta.sin();
        let sin_double = (2.0 * theta).sin();
        let a = cos * cos / (2.0 * sigma_x.powi(2)) + sin * sin / (2.0 * sigma_y.powi(2));
        let b = -sin_double / (4.0 * sigma_x.powi(2)) + sin_double / (4.0 * sigma_y.powi(2));
        let c = sin * sin / (2.0 * sigma_x.powi(2)) + cos * cos / (2.0 * sigma_y.powi(2));
        self.pixels
            .chunks_exact(self.width)
            .enumerate()
            .flat_map(move |(row, pixels)| {
                let y = row as f64 - center_y;
                let yy = c * y.powi(2);
                pixels.iter().enumerate().map(move |(column, pixel)| {
                    let x = column as f64 - center_x;
                    let value =
                        offset + amplitude * (-(a * x.powi(2) + 2.0 * b * x * y + yy)).exp();
                    value - pixel
                })
            })
    }
}

impl LeastSquaresProblem<f64, Dyn, U7> for GaussianFit<'_> {
    type ParameterStorage = Owned<f64, U7>;
    type ResidualStorage = Owned<f64, Dyn>;
    type JacobianStorage = Owned<f64, Dyn, U7>;

    fn set_params(&mut self, parameters: &FitParameters) {
        self.parameters = *parameters;
    }

    fn params(&self) -> FitParameters {
        self.parameters
    }

    fn residuals(&self) -> Option<DVector<f64>> {
        if (self.cancelled)() || self.evaluations.get() >= MAX_FIT_EVALUATIONS {
            return None;
        }
        self.evaluations.set(self.evaluations.get() + 1);
        Some(DVector::from_iterator(
            self.pixels.len(),
            self.residual_values(&self.parameters),
        ))
    }

    fn jacobian(&self) -> Option<Jacobian> {
        if (self.cancelled)() || self.evaluations.get() >= MAX_FIT_EVALUATIONS {
            return None;
        }
        let baseline: Vec<_> = self.residual_values(&self.parameters).collect();
        let mut matrix = Jacobian::zeros_generic(Dyn(self.pixels.len()), U7);
        for index in 0..PARAMETERS {
            // SciPy's lmdif uses forward differences with a parameter-relative step.
            let mut parameters = self.parameters;
            let mut step = DIFFERENCE_STEP * parameters[index].abs();
            if step == 0.0 {
                step = DIFFERENCE_STEP;
            }
            parameters[index] += step;
            for ((target, value), reference) in matrix
                .column_mut(index)
                .iter_mut()
                .zip(self.residual_values(&parameters))
                .zip(&baseline)
            {
                *target = (value - reference) / step;
            }
        }
        self.evaluations.set(self.evaluations.get() + PARAMETERS);
        Some(matrix)
    }
}
