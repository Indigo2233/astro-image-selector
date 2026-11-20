//! Half-flux radius calculations for the original measurement library.

use std::cmp::Ordering;

use crate::port::{Candidate, MeasurementConfig, median_and_mad, select_candidates};

#[derive(Clone, Copy, Debug)]
pub struct HfrConfig {
    pub measurement: MeasurementConfig,
    pub annulus_inner_radius: f64,
    pub annulus_outer_radius: f64,
}

impl Default for HfrConfig {
    fn default() -> Self {
        Self {
            measurement: MeasurementConfig::default(),
            annulus_inner_radius: 5.0,
            annulus_outer_radius: 8.0,
        }
    }
}

/// Optional brightness columns in the original source catalog.
#[derive(Clone, Copy)]
pub enum HfrCatalog<'a> {
    Peak(&'a [Candidate]),
    Flux {
        positions: &'a [[f64; 2]],
        flux: &'a [f64],
    },
    Positions(&'a [[f64; 2]]),
}

/// Measures one star using the cutout's edge pixels to estimate its background.
pub fn calculate_hfr(
    pixels: &[f64],
    width: usize,
    height: usize,
    x: f64,
    y: f64,
    config: &HfrConfig,
) -> Option<f64> {
    let radius = config.measurement.fit_radius;
    let image = Image::new(pixels, width, height)?;
    let center = image.rounded_center([x, y], radius)?;
    let distances = sorted_radii(radius);
    image.measure_with_edge_background(center, config, &distances)
}

/// Returns the median half-flux radius and its twice-radius FWHM estimate.
pub fn calculate_image_hfr(
    pixels: &[f64],
    width: usize,
    height: usize,
    sources: HfrCatalog<'_>,
    config: &HfrConfig,
) -> (f64, f64) {
    let Some(image) = Image::new(pixels, width, height) else {
        return (0.0, 0.0);
    };
    let radius = config.measurement.fit_radius;
    if !image.fits_radius(radius) {
        return (0.0, 0.0);
    }
    let positions = select_positions(sources, &config.measurement);
    if positions.is_empty() {
        return (0.0, 0.0);
    }
    let distances = sorted_radii(radius);
    let inner = config.annulus_inner_radius;
    let outer = config.annulus_outer_radius;
    if !(outer > inner && inner > 0.0)
        || positions
            .iter()
            .flatten()
            .any(|coordinate| !coordinate.is_finite())
    {
        if !outer.is_finite() {
            return (0.0, 0.0);
        }
        let fallback_radius = radius.max((outer.trunc() + 2.0) as usize);
        return summarize(
            positions
                .into_iter()
                .filter_map(|position| image.rounded_center(position, fallback_radius))
                .filter_map(|center| {
                    image.measure_with_edge_background(center, config, &distances)
                }),
        );
    }

    summarize(positions.into_iter().filter_map(|position| {
        let background = image.annulus_background(position, inner, outer)?;
        let center = image.rounded_center(position, radius)?;
        image.measure(center, radius, background, &distances)
    }))
}

/// Returns the median half-flux radius and its twice-radius FWHM estimate,
/// using the cutout edges and rejecting saturation in each star's center.
pub fn calculate_image_hfr_vectorized(
    pixels: &[f64],
    width: usize,
    height: usize,
    sources: HfrCatalog<'_>,
    config: &HfrConfig,
) -> (f64, f64) {
    let Some(image) = Image::new(pixels, width, height) else {
        return (0.0, 0.0);
    };
    let radius = config.measurement.fit_radius;
    if !image.fits_radius(radius) {
        return (0.0, 0.0);
    }
    let positions = select_positions(sources, &config.measurement);
    if positions.is_empty() {
        return (0.0, 0.0);
    }
    let distances = sorted_radii(radius);
    summarize(
        positions
            .into_iter()
            .filter_map(|position| image.rounded_center(position, radius))
            .filter_map(|center| image.measure_with_edge_background(center, config, &distances)),
    )
}

fn select_positions(sources: HfrCatalog<'_>, config: &MeasurementConfig) -> Vec<[f64; 2]> {
    match sources {
        HfrCatalog::Peak(peaks) => select_candidates(peaks, config)
            .into_iter()
            .map(|candidate| [candidate.x, candidate.y])
            .collect(),
        HfrCatalog::Flux { positions, flux } => {
            if positions.len() != flux.len() {
                return Vec::new();
            }
            if positions.len() <= config.max_stars {
                return positions.to_vec();
            }
            let mut indices: Vec<_> = (0..positions.len()).collect();
            indices.sort_by(|&a, &b| {
                flux[a]
                    .is_nan()
                    .cmp(&flux[b].is_nan())
                    .then_with(|| flux[a].partial_cmp(&flux[b]).unwrap_or(Ordering::Equal))
            });
            indices
                .into_iter()
                .rev()
                .take(config.max_stars)
                .map(|index| positions[index])
                .collect()
        }
        HfrCatalog::Positions(positions) => {
            positions[..positions.len().min(config.max_stars)].to_vec()
        }
    }
}

struct Image<'a> {
    pixels: &'a [f64],
    width: usize,
    height: usize,
}

impl<'a> Image<'a> {
    fn new(pixels: &'a [f64], width: usize, height: usize) -> Option<Self> {
        (width.checked_mul(height) == Some(pixels.len()) && !pixels.is_empty()).then_some(Self {
            pixels,
            width,
            height,
        })
    }

    fn fits_radius(&self, radius: usize) -> bool {
        radius <= (self.width.min(self.height) - 1) / 2
    }

    fn rounded_center(&self, [x, y]: [f64; 2], radius: usize) -> Option<[usize; 2]> {
        if !self.fits_radius(radius) {
            return None;
        }
        let (x, y) = (x.round_ties_even(), y.round_ties_even());
        (x >= radius as f64
            && x < (self.width - radius) as f64
            && y >= radius as f64
            && y < (self.height - radius) as f64)
            .then_some([x as usize, y as usize])
    }

    fn row(&self, y: usize) -> &[f64] {
        &self.pixels[y * self.width..(y + 1) * self.width]
    }

    fn annulus_background(&self, [x, y]: [f64; 2], inner: f64, outer: f64) -> Option<f64> {
        if x - outer < 0.0
            || y - outer < 0.0
            || x + outer >= self.width as f64
            || y + outer >= self.height as f64
        {
            return None;
        }
        let left = (x - outer).ceil() as usize;
        let right = (x + outer).floor() as usize;
        let top = (y - outer).ceil() as usize;
        let bottom = (y + outer).floor() as usize;
        let (inner, outer) = (inner.powi(2), outer.powi(2));
        let mut samples = Vec::new();
        for row in top..=bottom {
            for col in left..=right {
                let distance = (row as f64 - y).powi(2) + (col as f64 - x).powi(2);
                let pixel = self.row(row)[col];
                if distance >= inner && distance < outer && !pixel.is_nan() {
                    samples.push(pixel);
                }
            }
        }
        (samples.len() >= 10).then(|| median_and_mad(&mut samples).0)
    }

    fn measure_with_edge_background(
        &self,
        center: [usize; 2],
        config: &HfrConfig,
        distances: &[(usize, usize, usize)],
    ) -> Option<f64> {
        if self.saturated(center, config.measurement.saturation_level) {
            return None;
        }
        let [x, y] = center;
        let radius = config.measurement.fit_radius;
        let mut samples = Vec::with_capacity(4 * (2 * radius + 1));
        samples.extend_from_slice(&self.row(y - radius)[x - radius..=x + radius]);
        samples.extend_from_slice(&self.row(y + radius)[x - radius..=x + radius]);
        samples.extend((y - radius..=y + radius).map(|row| self.row(row)[x - radius]));
        samples.extend((y - radius..=y + radius).map(|row| self.row(row)[x + radius]));
        let background = median_and_mad(&mut samples).0;
        self.measure(center, radius, background, distances)
    }

    fn saturated(&self, [x, y]: [usize; 2], saturation: f64) -> bool {
        // NumPy's negative slice starts are relative to the end of the axis.
        let left = x.checked_sub(1).unwrap_or(self.width - 1);
        let top = y.checked_sub(1).unwrap_or(self.height - 1);
        let right = (x + 2).min(self.width);
        let bottom = (y + 2).min(self.height);
        if left >= right {
            return false;
        }
        (top..bottom).any(|row| {
            self.row(row)[left..right]
                .iter()
                .any(|&pixel| pixel >= saturation)
        })
    }

    fn measure(
        &self,
        [x, y]: [usize; 2],
        radius: usize,
        background: f64,
        distances: &[(usize, usize, usize)],
    ) -> Option<f64> {
        let left = x - radius;
        let top = y - radius;
        let mut cumulative = Vec::with_capacity(distances.len());
        let mut total = 0.0;
        for &(_, row, col) in distances {
            let flux = self.row(top + row)[left + col] - background;
            total += if flux < 0.0 { 0.0 } else { flux };
            cumulative.push(total);
        }
        if total <= 1e-6 {
            return None;
        }
        let half = total / 2.0;
        let index = cumulative.partition_point(|&sample| {
            // NumPy's searchsorted orders NaNs after numeric values.
            sample < half || (half.is_nan() && !sample.is_nan())
        });
        Some(
            distances
                .get(index)
                .map_or(radius as f64, |&(distance, _, _)| (distance as f64).sqrt()),
        )
    }
}

fn sorted_radii(radius: usize) -> Vec<(usize, usize, usize)> {
    let size = 2 * radius + 1;
    let mut distances: Vec<_> = (0..size)
        .flat_map(|row| {
            (0..size).map(move |col| {
                (
                    row.abs_diff(radius).pow(2) + col.abs_diff(radius).pow(2),
                    row,
                    col,
                )
            })
        })
        .collect();
    distances.sort_unstable();
    distances
}

fn summarize(values: impl Iterator<Item = f64>) -> (f64, f64) {
    let mut values = values.collect::<Vec<_>>();
    let median = median_and_mad(&mut values).0;
    (median, 2.0 * median)
}
