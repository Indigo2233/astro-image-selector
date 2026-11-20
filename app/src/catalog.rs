use std::path::{Path, PathBuf};

use serde::Serialize;

#[derive(Clone, Debug, Serialize)]
pub struct AnalysisResult {
    pub filename: PathBuf,
    pub filter_type: String,
    pub eccentricity: f64,
    pub fwhm: f64,
    pub snr: f64,
    pub star_count: usize,
    pub mean_brightness: f64,
}

impl AnalysisResult {
    pub fn unreadable(file: &Path) -> Self {
        Self {
            filename: file.to_path_buf(),
            filter_type: "Unknown".into(),
            eccentricity: 1.0,
            fwhm: 0.0,
            snr: 0.0,
            star_count: 0,
            mean_brightness: 0.0,
        }
    }
}

pub fn normalize_filter(name: &str) -> String {
    if name.is_empty() || name.eq_ignore_ascii_case("unknown") {
        return "Unknown".into();
    }
    let name = name.trim().to_uppercase();
    match name.as_str() {
        "LUMINANCE" => "L",
        "RED" => "R",
        "GREEN" => "G",
        "BLUE" => "B",
        "SULFUR" => "S",
        "HYDROGEN" => "H",
        "OXYGEN" => "O",
        "HALPHA" | "H-ALPHA" => "Ha",
        "O-III" => "OIII",
        "S-II" => "SII",
        _ => return name,
    }
    .into()
}

pub fn threshold_matches<'a>(
    results: &'a [AnalysisResult],
    metric: Metric,
    limit: f64,
    filter: Option<&'a str>,
) -> impl Iterator<Item = &'a Path> {
    results
        .iter()
        .filter(move |result| {
            filter.is_none_or(|name| result.filter_type == name) && metric.rejects(result, limit)
        })
        .map(|result| result.filename.as_path())
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Metric {
    Eccentricity,
    #[default]
    Fwhm,
    Snr,
    Stars,
    Brightness,
}

impl Metric {
    pub const ALL: [Self; 5] = [
        Self::Eccentricity,
        Self::Fwhm,
        Self::Snr,
        Self::Stars,
        Self::Brightness,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Self::Eccentricity => "椭率",
            Self::Fwhm => "FWHM",
            Self::Snr => "SNR",
            Self::Stars => "星点数量",
            Self::Brightness => "平均亮度",
        }
    }

    pub fn value(self, image: &AnalysisResult) -> f64 {
        match self {
            Self::Eccentricity => image.eccentricity,
            Self::Fwhm => image.fwhm,
            Self::Snr => image.snr,
            Self::Stars => image.star_count as f64,
            Self::Brightness => image.mean_brightness,
        }
    }

    pub fn plot_value(self, image: &AnalysisResult) -> f64 {
        let value = self.value(image);
        if value < 0.0 { 0.0 } else { value }
    }

    pub fn higher_is_better(self) -> bool {
        matches!(self, Self::Snr | Self::Stars)
    }

    pub fn rejects(self, image: &AnalysisResult, limit: f64) -> bool {
        let value = self.plot_value(image);
        value.is_finite()
            && if self.higher_is_better() {
                value < limit
            } else {
                value > limit
            }
    }

    pub fn limits(self) -> (f64, f64) {
        match self {
            Self::Eccentricity => (0.3, 1.0),
            Self::Fwhm => (3.5, 20.0),
            Self::Snr => (50.0, 3000.0),
            Self::Stars => (200.0, 5000.0),
            Self::Brightness => (5000.0, 30000.0),
        }
    }

    pub fn average_threshold(self, results: &[AnalysisResult]) -> Option<f64> {
        let (sum, count) = results
            .iter()
            .map(|result| self.value(result))
            .filter(|&value| value.is_finite() && value > 0.0)
            .fold((0.0, 0), |(sum, count), value| (sum + value, count + 1));
        (count > 0).then(|| (sum / f64::from(count)).min(self.limits().1))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn threshold_comparisons_are_strict_and_ignore_nonfinite_values() {
        let mut result = AnalysisResult::unreadable(Path::new("frame.fit"));
        result.fwhm = 3.5;
        result.snr = 48.0;

        assert!(!Metric::Fwhm.rejects(&result, 3.5));
        assert!(Metric::Fwhm.rejects(&result, 3.49));
        assert!(!Metric::Snr.rejects(&result, 48.0));
        assert!(Metric::Snr.rejects(&result, 48.01));
        result.snr = f64::NAN;
        assert!(!Metric::Snr.rejects(&result, 48.0));
    }

    #[test]
    fn normalize_existing_fits_filter_names() {
        assert_eq!(normalize_filter("H-ALPHA"), "Ha");
        assert_eq!(normalize_filter(" ha "), "HA");
        assert_eq!(normalize_filter(""), "Unknown");
        assert_eq!(normalize_filter(" unknown "), "UNKNOWN");
        assert_eq!(normalize_filter("o-iii"), "OIII");
    }

    #[test]
    fn threshold_selection_is_scoped_to_the_selected_filter() {
        let mut ha = AnalysisResult::unreadable(Path::new("a.fit"));
        let mut oiii = AnalysisResult::unreadable(Path::new("b.fit"));
        ha.filter_type = "Ha".into();
        ha.fwhm = 5.1;
        oiii.filter_type = "OIII".into();
        oiii.fwhm = 5.5;
        let results = [ha, oiii];
        assert_eq!(
            threshold_matches(&results, Metric::Fwhm, 5.0, Some("OIII")).collect::<Vec<_>>(),
            [Path::new("b.fit")]
        );
        assert_eq!(
            threshold_matches(&results, Metric::Fwhm, 5.0, None).count(),
            2
        );
    }
}
