use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, Eq, Serialize)]
pub enum AnalysisStatus {
    #[default]
    Success,
    ReadFailed,
    InvalidFilter,
    InvalidPixels,
    NoStars,
}

impl AnalysisStatus {
    pub fn label(self) -> &'static str {
        match self {
            Self::Success => "成功",
            Self::ReadFailed => "读取失败",
            Self::InvalidFilter => "滤镜头无效",
            Self::InvalidPixels => "像素无效",
            Self::NoStars => "未检出星点",
        }
    }

    pub fn is_success(self) -> bool {
        self == Self::Success
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct AnalysisResult {
    pub filename: PathBuf,
    #[serde(default)]
    pub status: AnalysisStatus,
    #[serde(default)]
    pub error: Option<String>,
    pub filter_type: String,
    pub eccentricity: f64,
    pub fwhm: f64,
    pub snr: f64,
    pub star_count: usize,
    pub mean_brightness: f64,
}

impl AnalysisResult {
    pub fn failed(file: &Path, status: AnalysisStatus, error: impl Into<String>) -> Self {
        Self {
            filename: file.to_path_buf(),
            status,
            error: Some(error.into()),
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

    pub fn status(self, image: &AnalysisResult) -> Option<f64> {
        image.status.is_success().then(|| self.value(image))
    }

    pub fn plot_value(self, image: &AnalysisResult) -> f64 {
        let value = self.status(image).unwrap_or(f64::NAN);
        if value < 0.0 { 0.0 } else { value }
    }

    pub fn higher_is_better(self) -> bool {
        matches!(self, Self::Snr | Self::Stars)
    }

    pub fn rejects(self, image: &AnalysisResult, limit: f64) -> bool {
        let value = self.status(image).unwrap_or(f64::NAN);
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
            .filter_map(|result| self.status(result))
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
        let mut result =
            AnalysisResult::failed(Path::new("frame.fit"), AnalysisStatus::Success, "unused");
        result.status = AnalysisStatus::Success;
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
        let mut ha = AnalysisResult::failed(Path::new("a.fit"), AnalysisStatus::Success, "unused");
        let mut oiii =
            AnalysisResult::failed(Path::new("b.fit"), AnalysisStatus::Success, "unused");
        ha.status = AnalysisStatus::Success;
        oiii.status = AnalysisStatus::Success;
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

    #[test]
    fn failed_results_never_enter_charts_thresholds_or_average_thresholds() {
        let failed = AnalysisResult::failed(
            Path::new("failed.fits"),
            AnalysisStatus::ReadFailed,
            "unreadable",
        );
        assert!(Metric::Fwhm.plot_value(&failed).is_nan());
        assert!(!Metric::Fwhm.rejects(&failed, 0.0));
        assert_eq!(Metric::Fwhm.average_threshold(&[failed]), None);
    }
}
