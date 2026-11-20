use astro_gpu::port::Gpu;
use astro_measurements::port::{Candidate, Measurement, MeasurementConfig};

const SIDE: usize = 31;
const GRID: usize = 5;
const FIXTURE: &[u8] = include_bytes!("../../measurements/tests/desktop_cutouts.bin");

fn candidate(x: usize, y: usize, peak: f64) -> Candidate {
    Candidate {
        x: x as f64,
        y: y as f64,
        peak,
    }
}

fn desktop_image() -> (Vec<f64>, Vec<Candidate>) {
    let width = GRID * SIDE;
    let mut pixels = vec![0.0; width * width];
    let mut stars = Vec::new();
    for (index, cutout) in FIXTURE
        .as_chunks::<{ SIDE * SIDE * 4 }>()
        .0
        .iter()
        .enumerate()
    {
        let x = index % GRID * SIDE;
        let y = index / GRID * SIDE;
        for (row, row_bytes) in cutout.as_chunks::<{ SIDE * 4 }>().0.iter().enumerate() {
            let start = (y + row) * width + x;
            for (pixel, &bytes) in pixels[start..start + SIDE]
                .iter_mut()
                .zip(row_bytes.as_chunks::<4>().0)
            {
                *pixel = f64::from(f32::from_le_bytes(bytes));
            }
        }
        stars.push(candidate(x + SIDE / 2, y + SIDE / 2, 4000.));
    }
    (pixels, stars)
}

#[test]
#[ignore = "requires a hardware compute adapter"]
fn desktop_cutouts_run_on_hardware() {
    let gpu = Gpu::new().expect("hardware compute adapter");
    let (pixels, stars) = desktop_image();
    let width = SIDE * GRID;
    let config = MeasurementConfig::default();
    let actual = gpu.measure(&pixels, width, width, &stars, &config).unwrap();
    println!("{}: desktop {actual:?}", gpu.name());

    assert_matches_reference("desktop", actual);
    assert!(
        gpu.measure_cancellable(&pixels, width, width, &stars, &config, || true)
            .unwrap()
            .is_none()
    );
}

#[test]
#[ignore = "requires a hardware compute adapter"]
fn filters_and_combines_hardware_batches() {
    let gpu = Gpu::new().expect("hardware compute adapter");
    let width = 3 * SIDE;
    let mut pixels = vec![1000.0; width * SIDE];
    for y in 0..SIDE {
        for x in 0..SIDE {
            let dx = x as f64 - 15.0;
            let dy = y as f64 - 15.0;
            pixels[y * width + SIDE + x] += 1450.0 * (-(dx * dx + dy * dy) / 9.68).exp();
            pixels[y * width + 2 * SIDE + x] +=
                2400.0 * (-(dx * dx / 11.52 + dy * dy / 5.12)).exp();
        }
    }
    pixels[30 * width + SIDE + 30] = f64::INFINITY;
    pixels[30 * width + 2 * SIDE + 30] = f64::NAN;
    let flat = candidate(15, 15, 1000.);
    let circular = candidate(46, 15, 1450.);
    let ellipse = candidate(77, 15, 2400.);
    let mut stars = vec![flat; 256];
    stars.extend([
        circular,
        ellipse,
        candidate(77, 15, 65_000.0),
        candidate(92, 15, 40.0),
    ]);
    let config = MeasurementConfig {
        max_stars: 256 + 4,
        ..Default::default()
    };
    for (label, candidates) in [
        ("mixed", stars.as_slice()),
        ("circle", &[circular]),
        ("ellipse", &[ellipse]),
        ("flat", &[flat]),
    ] {
        let actual = gpu
            .measure(&pixels, width, SIDE, candidates, &config)
            .unwrap();
        assert_matches_reference(label, actual);
    }
}

fn assert_matches_reference(name: &str, actual: Option<Measurement>) {
    let mut record = include_str!("fixtures.txt")
        .lines()
        .map(str::split_whitespace)
        .find(|fields| fields.clone().next() == Some(name))
        .expect("CuPy reference case")
        .skip(1);
    let count = record.next().unwrap().parse::<usize>().unwrap();
    if count == 0 {
        assert_eq!(actual, None, "{name}");
        return;
    }
    let fwhm = record.next().unwrap().parse::<f64>().unwrap();
    let eccentricity = record.next().unwrap().parse::<f64>().unwrap();
    let actual = actual.expect("valid measurements");
    assert_eq!(actual.used_stars, count, "{name}");
    assert!(
        (actual.fwhm - fwhm).abs() <= 1e-4,
        "{name}: {actual:?}, expected FWHM {fwhm}"
    );
    assert!(
        (actual.eccentricity - eccentricity).abs() <= 1e-4,
        "{name}: {actual:?}, expected eccentricity {eccentricity}"
    );
}
