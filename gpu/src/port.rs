use std::{
    sync::{Arc, Mutex, TryLockError, mpsc},
    thread,
    time::{Duration, Instant},
};

use astro_measurements::port::{
    Candidate, Measurement, MeasurementConfig, median_and_mad, select_candidates,
};
use wgpu::util::DeviceExt;

const BATCH_SIZE: usize = 256;
const GPU_TIMEOUT: Duration = Duration::from_secs(3);
const POLL_INTERVAL: Duration = Duration::from_millis(10);
type GpuMetrics = [f32; 4];

pub struct Gpu {
    name: String,
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bindings: wgpu::BindGroupLayout,
    fault: Arc<Mutex<Option<String>>>,
    submission_lock: Mutex<()>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PreferredGpu {
    Discrete(String),
    Integrated(String),
    Unavailable,
}

impl PreferredGpu {
    pub fn use_gpu(&self) -> bool {
        matches!(self, Self::Discrete(_))
    }

    pub fn device_name(&self) -> Option<&str> {
        match self {
            Self::Discrete(name) | Self::Integrated(name) => Some(name),
            Self::Unavailable => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct AdapterSummary {
    name: String,
    integrated: bool,
    dx12: bool,
}

pub fn preferred_gpu() -> PreferredGpu {
    let backends = wgpu::Backends::DX12 | wgpu::Backends::VULKAN;
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends,
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });
    let adapters = pollster::block_on(instance.enumerate_adapters(backends))
        .into_iter()
        .filter_map(|adapter| {
            let info = adapter.get_info();
            let integrated = match info.device_type {
                wgpu::DeviceType::DiscreteGpu => false,
                wgpu::DeviceType::IntegratedGpu => true,
                _ => return None,
            };
            Some(AdapterSummary {
                name: format!("{} ({})", info.name, info.backend),
                integrated,
                dx12: info.backend == wgpu::Backend::Dx12,
            })
        })
        .collect();
    select_preferred_gpu(adapters)
}

fn select_preferred_gpu(mut adapters: Vec<AdapterSummary>) -> PreferredGpu {
    adapters.sort_by_key(|adapter| (adapter.integrated, !adapter.dx12));
    match adapters.into_iter().next() {
        Some(adapter) if adapter.integrated => PreferredGpu::Integrated(adapter.name),
        Some(adapter) => PreferredGpu::Discrete(adapter.name),
        None => PreferredGpu::Unavailable,
    }
}

impl Gpu {
    pub fn new() -> Result<Self, String> {
        let backends = wgpu::Backends::DX12 | wgpu::Backends::VULKAN;
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends,
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let mut candidates: Vec<_> = pollster::block_on(instance.enumerate_adapters(backends))
            .into_iter()
            .filter_map(|adapter| {
                let info = adapter.get_info();
                let integrated = match info.device_type {
                    wgpu::DeviceType::DiscreteGpu => false,
                    wgpu::DeviceType::IntegratedGpu => true,
                    _ => return None,
                };
                let is_dx12 = info.backend == wgpu::Backend::Dx12;
                Some(((integrated, is_dx12), adapter, info))
            })
            .collect();
        candidates.sort_by_key(|(priority, _, _)| *priority);

        let mut failures = Vec::new();
        for (_, adapter, info) in candidates {
            let name = format!("{} ({})", info.name, info.backend);
            match Self::open(&adapter, name.clone()) {
                Ok(gpu) => return Ok(gpu),
                Err(error) => failures.push(format!("{name}: {error}")),
            }
        }
        if failures.is_empty() {
            Err("No hardware compute adapter found".to_owned())
        } else {
            Err(failures.join(" | "))
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn measure(
        &self,
        pixels: &[f64],
        width: usize,
        height: usize,
        stars: &[Candidate],
        config: &MeasurementConfig,
    ) -> Result<Option<Measurement>, String> {
        self.measure_cancellable(pixels, width, height, stars, config, || false)
    }

    /// Returns Ok(None) when no stars are suitable or the caller cancels the work.
    pub fn measure_cancellable(
        &self,
        pixels: &[f64],
        width: usize,
        height: usize,
        stars: &[Candidate],
        config: &MeasurementConfig,
        cancelled: impl Fn() -> bool,
    ) -> Result<Option<Measurement>, String> {
        if stars.is_empty() || cancelled() {
            return Ok(None);
        }
        let Some(shape) = CutoutShape::new(width, height, pixels.len(), config.fit_radius) else {
            return Ok(None);
        };
        let candidate_stars = select_candidates(stars, config);
        if candidate_stars.is_empty() {
            return Ok(None);
        }
        self.check_fault()?;
        let limits = self.device.limits();
        let max_bytes = limits
            .max_storage_buffer_binding_size
            .min(limits.max_buffer_size);
        let cutout_bytes = shape.area as u64 * size_of::<f32>() as u64;
        let batch_size = (max_bytes / cutout_bytes)
            .min(limits.max_compute_workgroups_per_dimension as u64)
            .min(BATCH_SIZE as u64) as usize;
        if batch_size == 0 {
            return Err("Cutout exceeds GPU buffer size limit".to_owned());
        }

        let mut values = Vec::new();
        let mut eccentricities = Vec::new();
        let mut batch_error = None;
        for stars in candidate_stars.chunks(batch_size) {
            if cancelled() {
                return Ok(None);
            }
            let batch = Cutouts::prepare(pixels, width, height, &shape, stars);
            if batch.backgrounds.is_empty() {
                continue;
            }
            match self.compute(&batch, &shape, &cancelled) {
                Ok(Some(rows)) => {
                    for [flux, fwhm, eccentricity, _] in rows {
                        if flux > 1e-6
                            && fwhm > 0.5
                            && fwhm < shape.side as f32 / 2.0
                            && (0.0..=1.0).contains(&eccentricity)
                        {
                            values.push(f64::from(fwhm));
                            eccentricities.push(f64::from(eccentricity));
                        }
                    }
                }
                Ok(None) => return Ok(None),
                Err(error) => {
                    batch_error.get_or_insert(error);
                }
            }
        }
        if cancelled() {
            return Ok(None);
        }
        if values.is_empty() {
            return batch_error.map_or(Ok(None), Err);
        }
        Ok(Some(Measurement {
            used_stars: values.len(),
            fwhm: median_and_mad(&mut values).0,
            eccentricity: median_and_mad(&mut eccentricities).0,
        }))
    }

    fn open(adapter: &wgpu::Adapter, name: String) -> Result<Self, String> {
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("AstroImageSelector measurements"),
            ..Default::default()
        }))
        .map_err(|error| error.to_string())?;

        let fault = Arc::new(Mutex::new(None));
        let lost_fault = Arc::clone(&fault);
        device.set_device_lost_callback(move |reason, _| {
            if let Ok(mut fault) = lost_fault.lock() {
                fault.get_or_insert_with(|| format!("GPU device lost ({reason:?})"));
            }
        });
        let unhandled_fault = Arc::clone(&fault);
        device.on_uncaptured_error(Arc::new(move |error: wgpu::Error| {
            if let Ok(mut fault) = unhandled_fault.lock() {
                fault.get_or_insert_with(|| error.to_string());
            }
        }));

        let (pipeline, bindings) = checked(&device, || {
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Star moments"),
                source: wgpu::ShaderSource::Wgsl(include_str!("moments.wgsl").into()),
            });
            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Star moments"),
                layout: None,
                module: &shader,
                entry_point: Some("measure"),
                compilation_options: Default::default(),
                cache: None,
            });
            let bindings = pipeline.get_bind_group_layout(0);
            (pipeline, bindings)
        })?;
        Ok(Self {
            name,
            device,
            queue,
            pipeline,
            bindings,
            fault,
            submission_lock: Mutex::new(()),
        })
    }

    fn check_fault(&self) -> Result<(), String> {
        let fault = self.fault.lock().map_err(|error| error.to_string())?;
        fault.clone().map_or(Ok(()), Err)
    }

    fn compute(
        &self,
        batch: &Cutouts,
        shape: &CutoutShape,
        cancelled: &dyn Fn() -> bool,
    ) -> Result<Option<Vec<GpuMetrics>>, String> {
        let _lock = loop {
            if cancelled() {
                return Ok(None);
            }
            match self.submission_lock.try_lock() {
                Ok(lock) => break lock,
                Err(TryLockError::WouldBlock) => thread::sleep(POLL_INTERVAL),
                Err(TryLockError::Poisoned(error)) => return Err(error.to_string()),
            }
        };
        self.check_fault()?;
        let (readback, submission) = checked(&self.device, || self.submit(batch, shape))?;
        let (sender, receiver) = mpsc::sync_channel(1);
        readback.map_async(wgpu::MapMode::Read, .., move |result| {
            let _ = sender.send(result);
        });
        let start = Instant::now();
        loop {
            if cancelled() {
                return Ok(None);
            }
            self.check_fault()?;
            match self.device.poll(wgpu::PollType::Wait {
                submission_index: Some(submission.clone()),
                timeout: Some(POLL_INTERVAL),
            }) {
                Ok(_) => break,
                Err(wgpu::PollError::Timeout) if start.elapsed() < GPU_TIMEOUT => continue,
                Err(error) => return Err(format!("GPU computation failed: {error}")),
            }
        }

        receiver
            .try_recv()
            .map_err(|error| format!("GPU readback callback: {error}"))?
            .map_err(|error| format!("GPU readback: {error}"))?;
        let mapped = readback
            .get_mapped_range(..)
            .map_err(|error| error.to_string())?;
        let values: &[GpuMetrics] = bytemuck::try_cast_slice(&mapped)
            .map_err(|error| format!("GPU readback layout: {error}"))?;
        let values = values.to_vec();
        drop(mapped);
        readback.unmap();
        Ok(Some(values))
    }

    fn submit(
        &self,
        batch: &Cutouts,
        shape: &CutoutShape,
    ) -> (wgpu::Buffer, wgpu::SubmissionIndex) {
        let device = &self.device;
        let shape_words = [shape.side, shape.area, 0, 0];
        let cutouts = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Cutout pixels"),
            contents: bytemuck::cast_slice(&batch.pixels),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let backgrounds = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Cutout backgrounds"),
            contents: bytemuck::cast_slice(&batch.backgrounds),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Cutout shape"),
            contents: bytemuck::cast_slice(&shape_words),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let size = (batch.backgrounds.len() * size_of::<GpuMetrics>()) as u64;
        let output = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Star measurements"),
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Star measurement readback"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let binding_buffers = [&cutouts, &backgrounds, &output, &uniform];
        let entries = binding_buffers.map(wgpu::Buffer::as_entire_binding);
        let [pixels, backgrounds, measurements, shape] = entries;
        let bindings = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Star moment buffers"),
            layout: &self.bindings,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: pixels,
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: backgrounds,
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: measurements,
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: shape,
                },
            ],
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Measure star moments"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bindings, &[]);
            pass.dispatch_workgroups(batch.backgrounds.len() as u32, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, size);
        let submission = self.queue.submit([encoder.finish()]);
        (readback, submission)
    }
}

fn checked<T>(device: &wgpu::Device, operation: impl FnOnce() -> T) -> Result<T, String> {
    let out_of_memory = device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
    let internal = device.push_error_scope(wgpu::ErrorFilter::Internal);
    let validation = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let result = operation();
    let errors = [validation.pop(), internal.pop(), out_of_memory.pop()];
    for pending in errors {
        if let Some(error) = pollster::block_on(pending) {
            return Err(error.to_string());
        }
    }
    Ok(result)
}

struct CutoutShape {
    side: u32,
    area: u32,
    radius: usize,
}

impl CutoutShape {
    fn new(width: usize, height: usize, len: usize, radius: usize) -> Option<Self> {
        let side = radius.checked_mul(2)?.checked_add(1)?;
        if side > width || side > height || width.checked_mul(height)? != len {
            return None;
        }
        let area = u32::try_from(side.checked_mul(side)?).ok()?;
        Some(Self {
            side: side as u32,
            area,
            radius,
        })
    }
}

struct Cutouts {
    pixels: Vec<f32>,
    backgrounds: Vec<f32>,
}

impl Cutouts {
    fn prepare(
        pixels: &[f64],
        width: usize,
        height: usize,
        shape: &CutoutShape,
        stars: &[Candidate],
    ) -> Self {
        let side = shape.side as usize;
        let radius = shape.radius;
        let mut batch = Self {
            pixels: Vec::with_capacity(stars.len() * shape.area as usize),
            backgrounds: Vec::with_capacity(stars.len()),
        };
        let mut border = Vec::with_capacity(4 * side);
        for star in stars {
            let (x, y) = (star.x.round_ties_even(), star.y.round_ties_even());
            if !(x >= radius as f64
                && x < (width - radius) as f64
                && y >= radius as f64
                && y < (height - radius) as f64)
            {
                continue;
            }
            let (x, y) = (x as usize, y as usize);
            let start = batch.pixels.len();
            for row in y - radius..=y + radius {
                let offset = row * width + x - radius;
                batch
                    .pixels
                    .extend(pixels[offset..offset + side].iter().map(|&value| {
                        let value = value as f32;
                        if value.is_finite() { value } else { 0.0 }
                    }));
            }
            let cutout = &batch.pixels[start..];
            border.clear();
            border.extend_from_slice(&cutout[..side]);
            border.extend_from_slice(&cutout[cutout.len() - side..]);
            for row in cutout.chunks_exact(side) {
                border.extend([row[0], row[side - 1]]);
            }
            border.sort_unstable_by(f32::total_cmp);
            let midpoint = border.len() / 2;
            batch
                .backgrounds
                .push((border[midpoint - 1] + border[midpoint]) / 2.0);
        }
        batch
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn discrete_adapter_enables_gpu_even_when_integrated_adapter_exists() {
        let preferred = select_preferred_gpu(vec![
            adapter("Intel UHD Graphics", true, false),
            adapter("NVIDIA RTX", false, true),
        ]);
        assert_eq!(preferred, PreferredGpu::Discrete("NVIDIA RTX".into()));
        assert!(preferred.use_gpu());
    }

    #[test]
    fn integrated_only_system_defaults_to_cpu() {
        let preferred = select_preferred_gpu(vec![adapter("Intel UHD Graphics", true, true)]);
        assert_eq!(
            preferred,
            PreferredGpu::Integrated("Intel UHD Graphics".into())
        );
        assert!(!preferred.use_gpu());
    }

    #[test]
    fn virtual_or_unavailable_adapters_default_to_cpu() {
        assert_eq!(select_preferred_gpu(Vec::new()), PreferredGpu::Unavailable);
        assert!(!PreferredGpu::Unavailable.use_gpu());
    }

    fn adapter(name: &str, integrated: bool, dx12: bool) -> AdapterSummary {
        AdapterSummary {
            name: name.into(),
            integrated,
            dx12,
        }
    }

    #[test]
    fn border_counts_corners_twice_and_sanitizes_nonfinite_samples() {
        let pixels = [
            f64::NAN,
            10.0,
            1.0,
            4.0,
            100.0,
            7.0,
            2.0,
            20.0,
            f64::INFINITY,
        ];
        let shape = CutoutShape::new(3, 3, pixels.len(), 1).unwrap();
        let center = Candidate {
            x: 1.,
            y: 1.,
            peak: 100.,
        };
        let outside = Candidate {
            x: 0.5,
            y: 1.,
            peak: 100.,
        };
        let batch = Cutouts::prepare(&pixels, 3, 3, &shape, &[center, outside]);
        assert_eq!(batch.backgrounds, [1.5]);
        assert_eq!(batch.pixels, [0., 10., 1., 4., 100., 7., 2., 20., 0.]);
    }

    #[test]
    fn validates_geometry_and_supports_a_single_pixel() {
        assert!(CutoutShape::new(8, 8, 64, 4).is_none());
        assert!(CutoutShape::new(8, 8, 63, 3).is_none());
        let shape = CutoutShape::new(1, 1, 1, 0).unwrap();
        let center = Candidate {
            x: 0.,
            y: 0.,
            peak: 5.,
        };
        let batch = Cutouts::prepare(&[5.], 1, 1, &shape, &[center]);
        assert_eq!(batch.backgrounds, [5.]);
        assert_eq!(batch.pixels, [5.]);
    }
}
