use astro_measurements::port::{
    Candidate, Measurement, MeasurementConfig, fit_star, measure, measure_cancellable,
    median_and_mad, select_candidates, snr,
};
use serde::Deserialize;

const FWHM_TOLERANCE: f64 = 1e-4;
const ECCENTRICITY_TOLERANCE: f64 = 1e-4;

#[derive(Deserialize)]
struct Reference {
    fit: Vec<FitCase>,
    measure: (f64, f64, usize),
    snr: f64,
    desktop: DesktopReference,
}

#[derive(Deserialize)]
struct FitCase {
    name: String,
    parameters: [f64; 10],
    expected: Option<(f64, f64)>,
}

#[derive(Deserialize)]
struct DesktopReference {
    expected: (f64, f64, usize),
}

fn reference() -> Reference {
    serde_json::from_str(include_str!("reference.json")).unwrap()
}

fn gaussian(parameters: [f64; 10]) -> Vec<f64> {
    let [
        width,
        height,
        amplitude,
        center_x,
        center_y,
        sigma_x,
        sigma_y,
        angle,
        offset,
        noise,
    ] = parameters;
    let (width, height) = (width as usize, height as usize);
    let (sin, cos) = angle.sin_cos();
    (0..width * height)
        .map(|index| {
            let (x, y) = ((index % width) as f64, (index / width) as f64);
            let u = (cos * (x - center_x) - sin * (y - center_y)) / sigma_x;
            let v = (sin * (x - center_x) + cos * (y - center_y)) / sigma_y;
            offset + amplitude * (-0.5 * (u * u + v * v)).exp() + noise * (x * 0.93 + y * 2.5).sin()
        })
        .collect()
}

fn assert_fit(label: &str, actual: (f64, f64), expected: (f64, f64)) {
    assert!(
        (actual.0 - expected.0).abs() <= FWHM_TOLERANCE,
        "{label}: FWHM {actual:?} vs. {expected:?}"
    );
    assert!(
        (actual.1 - expected.1).abs() <= ECCENTRICITY_TOLERANCE,
        "{label}: eccentricity {actual:?} vs. {expected:?}"
    );
}

fn assert_measurement(label: &str, actual: Measurement, expected: (f64, f64, usize)) {
    assert_eq!(actual.used_stars, expected.2, "{label}: fit count");
    assert_fit(
        label,
        (actual.fwhm, actual.eccentricity),
        (expected.0, expected.1),
    );
}

fn candidate(x: f64, y: f64, peak: f64) -> Candidate {
    Candidate { x, y, peak }
}

#[test]
fn gaussian_fits_match_regression_fixture() {
    for case in reference().fit {
        let [width, height, ..] = case.parameters;
        let image = gaussian(case.parameters);
        let actual = fit_star(&image, width as usize, height as usize);
        match (case.expected, actual) {
            (Some(expected), Some(actual)) => assert_fit(&case.name, actual, expected),
            (None, None) => (),
            mismatch => panic!("{}: fit result differs: {mismatch:?}", case.name),
        }
    }
}

#[test]
fn desktop_cutouts_match_fixture_aggregate() {
    let mut widths = Vec::new();
    let mut eccentricities = Vec::new();
    let cutouts = include_bytes!("desktop_cutouts.bin");
    let (chunks, remainder) = cutouts.as_chunks::<{ 31 * 31 * 4 }>();
    assert!(remainder.is_empty());
    for cutout in chunks {
        let pixels: Vec<f64> = cutout
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&bytes| f64::from(f32::from_le_bytes(bytes)))
            .collect();
        if let Some((width, eccentricity)) = fit_star(&pixels, 31, 31) {
            widths.push(width);
            eccentricities.push(eccentricity);
        }
    }
    let actual = Measurement {
        used_stars: widths.len(),
        fwhm: median_and_mad(&mut widths).0,
        eccentricity: median_and_mad(&mut eccentricities).0,
    };
    assert_measurement("desktop", actual, reference().desktop.expected);
}

#[test]
fn image_measurement_matches_reference_with_selection_and_snr() {
    let mut pixels = gaussian([93., 68., 1500., 20.5, 27.5, 1.9, 2.8, 0.62, 260., 3.]);
    let second = gaussian([93., 68., 2100., 50.5, 46.5, 1.5, 2.4, -0.36, 0., 0.]);
    let third = gaussian([93., 68., 1100., 79., 38., 2.7, 2.77, -0.27, 0., 0.]);
    for ((pixel, second), third) in pixels.iter_mut().zip(second).zip(third) {
        *pixel += second;
        *pixel += third;
    }
    let stars = [
        candidate(20.5, 27.5, 1500.),
        candidate(50.5, 46.5, 2100.),
        candidate(79., 38., 1100.),
        candidate(6., 20., 6900.),
        candidate(35., 35., 65000.),
    ];
    let config = MeasurementConfig {
        fit_radius: 12,
        max_stars: 3,
        ..Default::default()
    };
    let actual = measure(&pixels, 93, 68, &stars, &config);
    let expected = reference();
    assert_measurement("combined", actual, expected.measure);
    assert!((snr(&pixels) - expected.snr).abs() <= 1e-8 * expected.snr.abs().max(1.0));
}

#[test]
fn shape_invalid_pixels_empty_and_cancelled_return_deterministic_results() {
    let mut pixels = vec![12.; 31 * 31];
    let config = MeasurementConfig::default();
    let stars = [candidate(15., 15., 100.)];
    assert_eq!(
        measure(&pixels, 32, 31, &stars, &config),
        Measurement::default()
    );
    assert_eq!(
        measure(&pixels, 31, 31, &[], &config),
        Measurement::default()
    );
    assert!(measure_cancellable(&pixels, 31, 31, &stars, &config, || true).is_none());
    pixels[99] = f64::NAN;
    assert!(fit_star(&pixels, 31, 31).is_none());
    assert_eq!(snr(&pixels), 0.0);
    assert_eq!(snr(&[]), 0.0);
    assert_eq!(snr(&[1., 1., 1., 100.]), 0.0);
}

#[test]
fn selected_peaks_respect_limit_saturation_and_equal_values() {
    let stars: Vec<_> = (0..51)
        .map(|index| candidate(f64::from(index), 30., 4000.))
        .chain([candidate(60., 30., 65000.), candidate(61., 30., f64::NAN)])
        .collect();
    let chosen = select_candidates(&stars, &MeasurementConfig::default());
    assert_eq!(
        chosen,
        stars[1..51].iter().rev().copied().collect::<Vec<_>>()
    );
    assert_eq!(median_and_mad(&mut [12., 5., 1., 4.]), (4.5, 3.));
}
