use std::path::Path;

use astro_frames::port::{
    Adjustment, StretchParams, load, load_for_analysis, render_auto_preview, render_preview,
    statistical_stretch, thumbnail_median,
};
use serde::Deserialize;

#[derive(Deserialize)]
struct Recording {
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    shape: Option<Vec<usize>>,
    #[serde(default)]
    pixels: Vec<Pixel>,
    #[serde(default)]
    metadata: Option<Metadata>,
    #[serde(default)]
    thumbnail_median: Option<Pixel>,
    #[serde(default)]
    auto: String,
    #[serde(default)]
    manual: Vec<Manual>,
    #[serde(default)]
    custom: Option<String>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum Pixel {
    Value(f64),
    Special(String),
}

impl Pixel {
    fn assert_equal(&self, actual: f64, label: &str) {
        match self {
            Self::Value(expected) => assert_eq!(actual.to_bits(), expected.to_bits(), "{label}"),
            Self::Special(expected) if expected == "nan" => {
                assert!(actual.is_nan(), "{label} should be NaN, got {actual}");
            }
            Self::Special(expected) if expected == "inf" => {
                assert_eq!(actual, f64::INFINITY, "{label}");
            }
            Self::Special(expected) if expected == "-inf" => {
                assert_eq!(actual, f64::NEG_INFINITY, "{label}");
            }
            Self::Special(value) => panic!("Unknown reference pixel {value}"),
        }
    }
}

#[derive(Deserialize)]
struct Metadata {
    filter: serde_json::Value,
    exposure: f64,
    object: String,
    date: String,
    shape: [usize; 2],
    data_range: String,
}

#[derive(Deserialize)]
struct Manual {
    adjustment: [f64; 3],
    bytes: String,
}

#[test]
fn fits_and_preview_match_fixtures() {
    let recording: Recording = serde_json::from_str(include_str!("cases.json")).unwrap();
    assert!(!recording.cases.is_empty());
    for case in recording.cases {
        check_case(case);
    }
}

fn check_case(case: Case) {
    let label = case.name.as_str();
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/reference")
        .join(format!("{label}.fits"));
    let analyzed = load_for_analysis(&path);
    let preview = load(&path);
    let Some(shape) = case.shape else {
        assert!(analyzed.is_err(), "{label}");
        assert!(preview.is_err(), "{label}");
        return;
    };
    let analysis = analyzed.unwrap_or_else(|error| panic!("{label}: {error}"));
    assert_eq!(analysis.shape, shape, "{label}");
    assert_eq!(analysis.pixels.len(), case.pixels.len(), "{label}");
    for (index, (actual, expected)) in analysis.pixels.iter().zip(&case.pixels).enumerate() {
        expected.assert_equal(*actual, &format!("{label} analysis pixel {index}"));
    }
    let Some(expected) = case.metadata else {
        assert!(preview.is_err(), "{label}");
        return;
    };
    let frame = preview.unwrap_or_else(|error| panic!("{label}: {error}"));
    assert_eq!(frame.shape, shape, "{label}");
    assert_eq!([frame.height, frame.width], expected.shape, "{label}");
    assert_eq!(frame.pixels.len(), frame.width * frame.height, "{label}");
    for (index, (actual, expected)) in frame.pixels.iter().zip(&case.pixels).enumerate() {
        expected.assert_equal(*actual, &format!("{label} preview pixel {index}"));
    }
    let filter = expected
        .filter
        .as_str()
        .map(str::to_owned)
        .unwrap_or_else(|| expected.filter.to_string());
    assert_eq!(frame.metadata.filter, filter, "{label}");
    assert_eq!(
        frame.metadata.analysis_filter().is_ok(),
        expected.filter.is_string(),
        "{label}"
    );
    assert_eq!(frame.metadata.exposure, expected.exposure, "{label}");
    assert_eq!(frame.metadata.object, expected.object, "{label}");
    assert_eq!(frame.metadata.date, expected.date, "{label}");
    assert_eq!(frame.metadata.data_range, expected.data_range, "{label}");
    case.thumbnail_median
        .unwrap()
        .assert_equal(thumbnail_median(&frame).unwrap(), label);
    let stretched = statistical_stretch(&frame.pixels, StretchParams::default());
    assert_pixels(&render_auto_preview(&stretched), &case.auto, label);
    for manual in case.manual {
        let [brightness, contrast, gamma] = manual.adjustment;
        let rendered = render_preview(
            &stretched,
            Adjustment {
                brightness,
                contrast,
                gamma,
            },
        );
        assert_pixels(
            &rendered,
            &manual.bytes,
            &format!("{label} {:?}", manual.adjustment),
        );
    }
    if let Some(custom) = case.custom {
        let stretched = statistical_stretch(
            &frame.pixels,
            StretchParams {
                target_median: 0.33,
                sigma_factor: 1.4,
                curve_boost: 0.23,
                max_iterations: 3,
                ..StretchParams::default()
            },
        );
        assert_pixels(&render_auto_preview(&stretched), &custom, label);
    }
}

fn assert_pixels(actual: &[u8], expected_hex: &str, label: &str) {
    assert_eq!(actual.len() * 2, expected_hex.len(), "{label}");
    for (index, actual) in actual.iter().enumerate() {
        let expected = u8::from_str_radix(&expected_hex[(index * 2)..(index * 2 + 2)], 16).unwrap();
        assert_eq!(*actual, expected, "{label} byte {index}");
    }
}
