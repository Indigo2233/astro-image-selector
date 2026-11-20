use std::fs::File;
use std::io::{BufRead, BufReader, Cursor, Read, Seek, SeekFrom};
use std::path::Path;

use bzip2::bufread::MultiBzDecoder;
use fitsrs::Pixels;
use fitsrs::card::{Card, CardBuf, Value};
use fitsrs::hdu::data::FitsRead;
use fitsrs::hdu::header::extension::image::Image;
use fitsrs::hdu::header::{Bitpix, Header};
use fitsrs::hdu::primary::consume_next_card;
use flate2::bufread::MultiGzDecoder;
use lzma_rust2::XzReader;
use zip::ZipArchive;

#[derive(Debug)]
pub struct Frame {
    pub width: usize,
    pub height: usize,
    /// Original FITS image shape, with the fastest axis last as in NumPy.
    pub shape: Vec<usize>,
    /// Row-major physical pixel values. See the selected loader for the plane count.
    pub pixels: Vec<f64>,
    pub metadata: Metadata,
}

#[derive(Debug)]
pub struct Metadata {
    pub filter: String,
    pub exposure: f64,
    pub object: String,
    pub date: String,
    pub data_range: String,
    analysis_filter: AnalysisFilter,
}

#[derive(Debug)]
enum AnalysisFilter {
    Text,
    Missing,
    Invalid,
}

impl Metadata {
    pub fn analysis_filter(&self) -> Result<&str, String> {
        match self.analysis_filter {
            AnalysisFilter::Text => Ok(&self.filter),
            AnalysisFilter::Missing => Ok("Unknown"),
            AnalysisFilter::Invalid => Err("FITS FILTER must be a string".to_owned()),
        }
    }

    pub fn is_flat(&self) -> bool {
        flat_name_match(&self.object)
    }
}

/// Load the primary HDU and use its first plane for the grayscale preview.
pub fn load(path: &Path) -> Result<Frame, String> {
    read_frame(path, false)
}

/// Load all primary-HDU pixels for analysis, including all planes in a cube.
pub fn load_for_analysis(path: &Path) -> Result<Frame, String> {
    read_frame(path, true)
}

fn read_frame(path: &Path, all_planes: bool) -> Result<Frame, String> {
    let context =
        |message: &str| format!("Failed to load FITS image {}: {message}", path.display());
    let file = File::open(path).map_err(|error| context(&error.to_string()))?;
    read_input(BufReader::new(file), all_planes).map_err(|error| context(&error))
}

fn read_input(mut reader: BufReader<File>, all_planes: bool) -> Result<Frame, String> {
    let magic = reader.fill_buf().map_err(|error| error.to_string())?;
    if magic.starts_with(b"\x1f\x8b\x08") {
        decode_compressed(MultiGzDecoder::new(reader), all_planes)
    } else if magic.starts_with(b"BZ") {
        decode_compressed(MultiBzDecoder::new(reader), all_planes)
    } else if magic.starts_with(b"\xfd7zXZ\0") {
        decode_compressed(XzReader::new(reader, true), all_planes)
    } else if magic.starts_with(b"PK\x03\x04") {
        let mut archive = ZipArchive::new(reader).map_err(|error| error.to_string())?;
        if archive.len() != 1 {
            return Err("A FITS ZIP archive must contain one file".to_owned());
        }
        let member = archive.by_index(0).map_err(|error| error.to_string())?;
        decode_compressed(member, all_planes)
    } else {
        decode_frame(reader, all_planes)
    }
}

fn decode_compressed(mut stream: impl Read, all_planes: bool) -> Result<Frame, String> {
    let mut data = Vec::new();
    stream
        .read_to_end(&mut data)
        .map_err(|error| error.to_string())?;
    decode_frame(Cursor::new(data), all_planes)
}

fn decode_frame<R: Read + Seek + std::fmt::Debug>(
    mut reader: R,
    all_planes: bool,
) -> Result<Frame, String> {
    let hdu = read_primary_header(&mut reader)?;
    let header = hdu.get_header();
    let image = header.get_xtension();
    let shape = image
        .get_naxis()
        .iter()
        .rev()
        .map(|&axis| usize::try_from(axis).map_err(|_| "FITS axis length exceeds platform size"))
        .collect::<Result<Vec<_>, _>>()?;
    if shape.is_empty() || shape.contains(&0) {
        return Err("Primary FITS HDU contains no pixels".to_owned());
    }
    if shape.len() < 2 && !all_planes {
        return Err("FITS preview requires at least two image axes".to_owned());
    }
    let width = shape[shape.len() - 1];
    let height = shape.iter().rev().nth(1).copied().unwrap_or(1);
    let pixel_count = shape
        .iter()
        .try_fold(1_usize, |count, &axis| count.checked_mul(axis))
        .ok_or("FITS image dimensions overflow")?;
    let plane_size = width * height;
    let bytes = pixel_count
        .checked_mul(image.get_bitpix().byte_size())
        .ok_or("FITS image byte size overflows")? as u64;
    let data_offset = hdu.get_data_unit_byte_offset();
    let file_size = reader
        .seek(SeekFrom::End(0))
        .map_err(|error| error.to_string())?;
    if bytes > file_size.saturating_sub(data_offset) {
        return Err("Primary FITS data is truncated".to_owned());
    }
    reader
        .seek(SeekFrom::Start(data_offset))
        .map_err(|error| error.to_string())?;

    let limit = if all_planes { pixel_count } else { plane_size };
    let scale = PixelScale::from_header(header);
    let input = reader.read_data_unit(header, data_offset).pixels();
    let mut pixels = Vec::new();
    pixels
        .try_reserve_exact(limit)
        .map_err(|error| error.to_string())?;
    match input {
        Pixels::U8(values) => pixels.extend(values.take(limit).map(|x| scale.integer(x.into()))),
        Pixels::I16(values) => pixels.extend(values.take(limit).map(|x| scale.integer(x.into()))),
        Pixels::I32(values) => pixels.extend(values.take(limit).map(|x| scale.integer(x.into()))),
        Pixels::I64(values) => pixels.extend(values.take(limit).map(|x| scale.integer(x))),
        Pixels::F32(values) => pixels.extend(values.take(limit).map(|x| scale.float32(x))),
        Pixels::F64(values) => pixels.extend(values.take(limit).map(|x| scale.float64(x))),
    }
    if pixels.len() != limit {
        return Err("Could not read the complete primary FITS image".to_owned());
    }
    let metadata = frame_metadata(header, &pixels[..plane_size], all_planes);
    Ok(Frame {
        width,
        height,
        shape,
        pixels,
        metadata,
    })
}

fn read_primary_header<R: Read + Seek + std::fmt::Debug>(
    reader: &mut R,
) -> Result<fitsrs::fits::HDU<Image>, String> {
    // Read the header separately so dimension arithmetic can be checked before
    // fitsrs calculates the data size from the header.
    let mut card: CardBuf = [0; 80];
    let mut bytes_read = 0;
    consume_next_card(reader, &mut card, &mut bytes_read).map_err(|error| error.to_string())?;
    let first = Card::try_from(&card).map_err(|error| error.to_string())?;
    if !matches!(
        &first,
        Card::Value { name, value: Value::Logical { value: true, .. } } if name == "SIMPLE"
    ) {
        return Err("Missing SIMPLE keyword in the primary FITS header".to_owned());
    }
    let mut cards = vec![first];
    loop {
        consume_next_card(reader, &mut card, &mut bytes_read).map_err(|error| error.to_string())?;
        let value = Card::try_from(&card)
            .unwrap_or_else(|_| Card::Undefined(String::from_utf8_lossy(&card).into_owned()));
        let end = value == Card::End;
        cards.push(value);
        if end {
            break;
        }
    }
    fitsrs::fits::HDU::new(reader, &mut bytes_read, cards).map_err(|error| error.to_string())
}

struct PixelScale {
    factor: f64,
    zero: f64,
    blank: Option<i64>,
    bitpix: Bitpix,
}

impl PixelScale {
    fn from_header(header: &Header<Image>) -> Self {
        Self {
            factor: header_number(header.get("BSCALE")).unwrap_or(1.0),
            zero: header_number(header.get("BZERO")).unwrap_or(0.0),
            blank: match header.get("BLANK") {
                Some(Value::Integer { value, .. }) => Some(*value),
                _ => None,
            },
            bitpix: header.get_xtension().get_bitpix(),
        }
    }

    fn integer(&self, value: i64) -> f64 {
        if self.factor == 1.0 {
            let unsigned_offset = match self.bitpix {
                Bitpix::U8 if self.zero == -128.0 => return (value - 128) as f64,
                Bitpix::I16 if self.zero == 32_768.0 => Some(1_u64 << 15),
                Bitpix::I32 if self.zero == 2_147_483_648.0 => Some(1_u64 << 31),
                Bitpix::I64 if self.zero == 9_223_372_036_854_775_808.0 => Some(1_u64 << 63),
                _ => None,
            };
            if let Some(offset) = unsigned_offset {
                return (value as u64).wrapping_add(offset) as f64;
            }
            if self.zero == 0.0 && self.blank.is_none() {
                return value as f64;
            }
        }
        // Astropy skips the BLANK substitution when the stored sentinel is 0.
        if self.blank.is_some_and(|blank| blank != 0 && blank == value) {
            return f64::NAN;
        }
        match self.bitpix {
            Bitpix::U8 | Bitpix::I16 => self.float32(value as f32),
            _ => self.float64(value as f64),
        }
    }

    fn float32(&self, mut value: f32) -> f64 {
        if self.factor != 1.0 {
            value *= self.factor as f32;
        }
        if self.zero != 0.0 {
            value += self.zero as f32;
        }
        f64::from(value)
    }

    fn float64(&self, mut value: f64) -> f64 {
        if self.factor != 1.0 {
            value *= self.factor;
        }
        if self.zero != 0.0 {
            value += self.zero;
        }
        value
    }
}

fn header_number(value: Option<&Value>) -> Option<f64> {
    match value {
        Some(Value::Float { value, .. }) => Some(*value),
        Some(Value::Integer { value, .. }) => Some(*value as f64),
        _ => None,
    }
}

fn header_text(value: Option<&Value>, default: &str) -> String {
    match value {
        Some(Value::String { value, .. }) | Some(Value::Invalid(value)) => value.clone(),
        Some(Value::Integer { value, .. }) => value.to_string(),
        Some(Value::Float { value, .. }) => format!("{value:?}"),
        Some(Value::Logical { value: true, .. }) => "True".to_owned(),
        Some(Value::Logical { value: false, .. }) => "False".to_owned(),
        _ => default.to_owned(),
    }
}

fn normalized_metadata(value: &str) -> String {
    value
        .trim()
        .trim_matches('"')
        .chars()
        .filter(|character| character.is_ascii_alphanumeric())
        .flat_map(char::to_uppercase)
        .collect()
}

fn frame_metadata(header: &Header<Image>, pixels: &[f64], all_planes: bool) -> Metadata {
    let filter = header.get("FILTER");
    let analysis_filter = match filter {
        Some(Value::String { .. }) => AnalysisFilter::Text,
        None
        | Some(Value::Integer { value: 0, .. })
        | Some(Value::Float { value: 0.0, .. })
        | Some(Value::Logical { value: false, .. }) => AnalysisFilter::Missing,
        _ => AnalysisFilter::Invalid,
    };
    let exposure = header.get("EXPTIME");
    let exposure = header_number(exposure)
        .or_else(|| {
            if let Some(Value::String { value, .. }) = exposure {
                value.trim().parse().ok()
            } else {
                None
            }
        })
        .unwrap_or(0.0);
    let (min, max) = if all_planes {
        (f64::NAN, f64::NAN)
    } else {
        pixel_range(pixels)
    };
    Metadata {
        filter: header_text(filter, "Unknown"),
        exposure,
        object: header_text(header.get("OBJECT"), "Unknown"),
        date: header_text(header.get("DATE-OBS"), ""),
        data_range: format!("[{}, {}]", format_pixel(min), format_pixel(max)),
        analysis_filter,
    }
}

/// Read only the primary header and classify common calibration-frame names.
pub fn is_flat_frame(path: &Path) -> bool {
    let file_name_is_flat = flat_name_match(
        path.file_stem()
            .and_then(|stem| stem.to_str())
            .unwrap_or_default(),
    );
    let header_is_flat = File::open(path)
        .ok()
        .and_then(|file| read_primary_header(&mut BufReader::new(file)).ok())
        .is_some_and(|hdu| {
            let header = hdu.get_header();
            ["IMAGETYP", "IMAGETYPE", "FRAME", "OBSTYPE", "OBSTYP"]
                .into_iter()
                .any(|key| flat_value_match(&header_text(header.get(key), "")))
        });
    file_name_is_flat || header_is_flat
}

fn flat_name_match(value: &str) -> bool {
    normalized_metadata(value).contains("FLAT")
}

fn flat_value_match(value: &str) -> bool {
    let normalized = normalized_metadata(value);
    let words: Vec<_> = value
        .split(|character: char| !character.is_ascii_alphanumeric())
        .filter(|word| !word.is_empty())
        .map(|word| word.to_ascii_uppercase())
        .collect();
    matches!(
        normalized.as_str(),
        "FLAT" | "FLATFIELD" | "FLATFRAME" | "SKYFLAT" | "TWILIGHTFLAT" | "DOME" | "DOMEFLAT"
    ) || words.iter().any(|word| word == "FLAT")
}

#[cfg(test)]
mod flat_tests {
    use super::*;
    use std::fs;

    #[test]
    fn recognizes_flat_headers_and_common_name_forms() {
        assert!(flat_value_match("FLAT"));
        assert!(flat_value_match("Light Flat"));
        assert!(flat_name_match("Calibration_Flats"));
        assert!(!flat_value_match("LIGHT"));
        assert!(!flat_name_match("NGC7000"));
    }

    #[test]
    fn header_identifies_a_flat_without_flat_in_the_filename() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("IMX411_0001.fits");
        let mut bytes = Vec::new();
        let header = [
            [
                "SIMPLE  =                    T",
                "BITPIX  =                    8",
            ],
            [
                "NAXIS   =                    2",
                "NAXIS1  =                    2",
            ],
            ["NAXIS2  =                    2", "IMAGETYP= 'FLAT    '"],
            ["END", ""],
        ];
        for mut block in header {
            for card in &mut block {
                let line = format!("{card:<80}");
                bytes.extend_from_slice(line.as_bytes());
            }
        }
        bytes.resize(2880, b' ');
        fs::write(&path, bytes).unwrap();
        assert!(is_flat_frame(&path));
    }
}

fn pixel_range(pixels: &[f64]) -> (f64, f64) {
    let (mut min, mut max) = (f64::INFINITY, f64::NEG_INFINITY);
    for &pixel in pixels {
        if pixel.is_nan() {
            return (f64::NAN, f64::NAN);
        }
        if pixel <= min {
            min = pixel;
        }
        if pixel >= max {
            max = pixel;
        }
    }
    (min, max)
}

fn format_pixel(value: f64) -> String {
    if value.is_nan() {
        "nan".to_owned()
    } else {
        format!("{value:.2}")
    }
}

#[derive(Clone, Copy, Debug)]
pub struct StretchParams {
    pub target_median: f64,
    pub sigma_factor: f64,
    pub curve_boost: f64,
    pub max_iterations: usize,
    pub convergence_threshold: f64,
}

impl Default for StretchParams {
    fn default() -> Self {
        Self {
            target_median: 0.25,
            sigma_factor: 2.7,
            curve_boost: 0.0,
            max_iterations: 1,
            convergence_threshold: 0.001,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Adjustment {
    pub brightness: f64,
    pub contrast: f64,
    pub gamma: f64,
}

impl Default for Adjustment {
    fn default() -> Self {
        Self {
            brightness: 0.0,
            contrast: 1.0,
            gamma: 1.0,
        }
    }
}

/// Return the float grayscale image used for automatic and manual preview.
pub fn statistical_stretch(pixels: &[f64], params: StretchParams) -> Vec<f64> {
    if pixels.is_empty() {
        return Vec::new();
    }
    let pixels = pixels.to_vec();
    let (min, max) = pixel_range(&pixels);
    if (max - min).abs() <= f64::EPSILON * max.abs().max(1.0) {
        return vec![0.0; pixels.len()];
    }
    let mut stretched: Vec<_> = if max > 1.0 {
        pixels.iter().map(|&pixel| pixel / max).collect()
    } else {
        pixels
    };
    let mut scratch = Vec::with_capacity(stretched.len());
    let target = params.target_median;
    for iteration in 0..params.max_iterations {
        let (min, _) = pixel_range(&stretched);
        let middle = median(&stretched, &mut scratch);
        let len = stretched.len() as f64;
        let mean = stretched.iter().sum::<f64>() / len;
        let deviation = (stretched
            .iter()
            .map(|pixel| {
                let diff = pixel - mean;
                diff * diff
            })
            .sum::<f64>()
            / len)
            .sqrt();
        let black_point = middle - params.sigma_factor * deviation;
        let black_point = if min > black_point { min } else { black_point };
        for pixel in &mut stretched {
            *pixel = ((*pixel - black_point) / (1.0 - black_point)).clamp(0.0, 1.0);
        }
        let current = median(&stretched, &mut scratch);
        for pixel in &mut stretched {
            let scaled = target * *pixel;
            let denominator = current * (target + *pixel - 1.0) - scaled;
            let denominator = if denominator.abs() < 1e-10 {
                1e-10
            } else {
                denominator
            };
            *pixel = (((current - 1.0) * target * *pixel) / denominator).clamp(0.0, 10.0);
        }
        if iteration + 1 < params.max_iterations
            && (median(&stretched, &mut scratch) - target).abs() < params.convergence_threshold
        {
            break;
        }
    }
    if params.curve_boost > 0.0 {
        boost_curve(&mut stretched, target, params.curve_boost);
    }
    for pixel in &mut stretched {
        *pixel = pixel.clamp(0.0, f64::INFINITY);
    }
    stretched
}

fn median(pixels: &[f64], scratch: &mut Vec<f64>) -> f64 {
    if pixels.iter().any(|pixel| pixel.is_nan()) {
        return f64::NAN;
    }
    scratch.clear();
    scratch.extend_from_slice(pixels);
    let middle = scratch.len() / 2;
    let even = scratch.len().is_multiple_of(2);
    let (lower, upper, _) = scratch.select_nth_unstable_by(middle, f64::total_cmp);
    if even {
        let lower = lower.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        (lower + *upper) / 2.0
    } else {
        *upper
    }
}

fn boost_curve(pixels: &mut [f64], target: f64, boost: f64) {
    let shadow = target * 0.5;
    let highlight = target * 2.0;
    let max_highlight = pixels.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    for pixel in pixels {
        if *pixel <= shadow {
            *pixel = (*pixel / shadow).powf(1.0 - boost * 0.3) * shadow;
        } else if *pixel <= highlight {
            let range = target * 1.5;
            *pixel = shadow + ((*pixel - shadow) / range).powf(1.0 - boost * 0.7) * range;
        } else if *pixel > highlight && max_highlight > highlight {
            let range = max_highlight - highlight;
            let normalized = (*pixel - highlight) / range;
            let enhanced = (normalized * boost * 2.0).ln_1p() / (boost * 2.0).ln_1p();
            *pixel = highlight + enhanced * range;
        }
    }
}

pub fn render_auto_preview(stretched: &[f64]) -> Vec<u8> {
    stretched.iter().map(|&pixel| preview_byte(pixel)).collect()
}

pub fn render_preview(stretched: &[f64], adjustment: Adjustment) -> Vec<u8> {
    if adjustment.gamma == 0.0 {
        return render_auto_preview(stretched);
    }
    let exponent = 1.0 / adjustment.gamma;
    stretched
        .iter()
        .map(|&pixel| {
            let brightness = pixel + adjustment.brightness;
            let contrast = (brightness - 0.5) * adjustment.contrast + 0.5;
            let adjusted = contrast.clamp(0.0, 1.0).powf(exponent);
            preview_byte(adjusted)
        })
        .collect()
}

fn preview_byte(pixel: f64) -> u8 {
    (pixel.clamp(0.0, 1.0) * 255.0) as u8
}

/// The background preview samples each eighth pixel and row in single precision.
pub fn thumbnail_median(frame: &Frame) -> Option<f64> {
    let mut thumbnail: Vec<_> = frame
        .pixels
        .chunks(frame.width.max(1))
        .take(frame.height)
        .step_by(8)
        .flat_map(|row| row.iter().step_by(8).map(|&pixel| pixel as f32))
        .collect();
    if thumbnail.is_empty() {
        return None;
    }
    if thumbnail.iter().any(|pixel| pixel.is_nan()) {
        return Some(f64::NAN);
    }
    let middle = thumbnail.len() / 2;
    let even = thumbnail.len().is_multiple_of(2);
    let (lower, upper, _) = thumbnail.select_nth_unstable_by(middle, f32::total_cmp);
    let median = if even {
        let lower = lower.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        (lower + *upper) / 2.0
    } else {
        *upper
    };
    Some(f64::from(median))
}
