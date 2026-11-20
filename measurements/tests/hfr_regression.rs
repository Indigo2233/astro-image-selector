use astro_measurements::{
    hfr::{
        HfrCatalog, HfrConfig, calculate_hfr, calculate_image_hfr, calculate_image_hfr_vectorized,
    },
    port::{Candidate, MeasurementConfig},
};
use serde::Deserialize;

#[derive(Deserialize)]
struct Case {
    name: String,
    shape: [usize; 2],
    sky: [f64; 3],
    gaussians: Vec<[f64; 5]>,
    catalog: Vec<[f64; 3]>,
    brightness: String,
    center: [f64; 2],
    overrides: Vec<(usize, usize, serde_json::Value)>,
    radius: usize,
    annulus: [f64; 2],
    saturation: f64,
    max_stars: usize,
    single_hfr: Option<f64>,
    batch_hfr: (f64, f64),
    vectorized_hfr: (f64, f64),
}

impl Case {
    fn image(&self) -> Vec<f64> {
        let [width, height] = self.shape;
        let [level, dx, dy] = self.sky;
        let mut pixels = (0..width * height)
            .map(|index| {
                let x = (index % width) as f64;
                let y = (index / width) as f64;
                let sky = level + x * dx + y * dy;
                self.gaussians
                    .iter()
                    .fold(sky, |pixel, &[cx, cy, amplitude, sx, sy]| {
                        let squared_radius = ((x - cx) / sx).powi(2) + ((y - cy) / sy).powi(2);
                        pixel + amplitude * (-0.5 * squared_radius).exp()
                    })
            })
            .collect::<Vec<_>>();
        for (x, y, value) in &self.overrides {
            pixels[y * width + x] = value.as_f64().unwrap_or_else(|| {
                assert_eq!(value.as_str(), Some("nan"));
                f64::NAN
            });
        }
        pixels
    }

    fn config(&self) -> HfrConfig {
        HfrConfig {
            measurement: MeasurementConfig {
                saturation_level: self.saturation,
                max_stars: self.max_stars,
                fit_radius: self.radius,
            },
            annulus_inner_radius: self.annulus[0],
            annulus_outer_radius: self.annulus[1],
        }
    }
}

#[test]
fn hfr_methods_match_reference() {
    let cases: Vec<Case> = serde_json::from_str(include_str!("hfr_reference.json")).unwrap();
    assert_eq!(cases.len(), 7);
    for case in cases {
        let [width, height] = case.shape;
        let [x, y] = case.center;
        let config = case.config();
        let image = case.image();
        let peaks: Vec<_> = case
            .catalog
            .iter()
            .map(|&[x, y, peak]| Candidate { x, y, peak })
            .collect();
        let positions: Vec<_> = case.catalog.iter().map(|&[x, y, _]| [x, y]).collect();
        let flux: Vec<_> = case.catalog.iter().map(|&[_, _, flux]| flux).collect();
        let sources = match case.brightness.as_str() {
            "peak" => HfrCatalog::Peak(&peaks),
            "flux" => HfrCatalog::Flux {
                positions: &positions,
                flux: &flux,
            },
            "positions" => HfrCatalog::Positions(&positions),
            _ => panic!("Unexpected source catalog: {}", case.brightness),
        };
        let single = calculate_hfr(&image, width, height, x, y, &config);
        match (single, case.single_hfr) {
            (Some(actual), Some(expected)) => {
                assert!((actual - expected).abs() < 1e-12, "{}", case.name)
            }
            (None, None) => {}
            mismatch => panic!("{}: single-star measurement {mismatch:?}", case.name),
        }
        let batch = calculate_image_hfr(&image, width, height, sources, &config);
        assert_close(batch, case.batch_hfr, &case.name);
        let vectorized = calculate_image_hfr_vectorized(&image, width, height, sources, &config);
        assert_close(vectorized, case.vectorized_hfr, &case.name);
    }
}

#[test]
fn rejects_empty_measurements() {
    let image = [1.0; 1600];
    let config = HfrConfig::default();
    let stars = HfrCatalog::Peak(&[]);
    assert_eq!(calculate_hfr(&image, 40, 40, 19.0, 19.0, &config), None);
    assert_eq!(calculate_hfr(&image, 40, 40, 5.0, 19.0, &config), None);
    assert_eq!(calculate_hfr(&image, 1, 40, 19.0, 19.0, &config), None);
    assert_eq!(
        calculate_image_hfr(&image, 40, 40, stars, &config),
        (0.0, 0.0)
    );
    assert_eq!(
        calculate_image_hfr_vectorized(&image, 40, 40, stars, &config),
        (0.0, 0.0)
    );
}

fn assert_close(actual: (f64, f64), expected: (f64, f64), label: &str) {
    assert!(
        (actual.0 - expected.0).abs() < 1e-12 && (actual.1 - expected.1).abs() < 1e-12,
        "{label}: expected {expected:?}, got {actual:?}",
    );
}
