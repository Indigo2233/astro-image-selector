use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::mpsc::{Receiver, Sender, channel};
use std::time::SystemTime;

use astro_frames::port::{
    Adjustment, Metadata, StretchParams, load, render_auto_preview, render_preview,
    statistical_stretch,
};

pub struct PreviewRequest {
    pub id: u64,
    pub path: PathBuf,
    pub adjustment: Adjustment,
    pub automatic: bool,
    pub refresh: bool,
    pub neighbors: Vec<PathBuf>,
    pub texture_limit: usize,
}

pub struct Preview {
    pub pixels: Vec<u8>,
    pub texture_size: [usize; 2],
    pub image_size: [usize; 2],
    pub metadata: Arc<Metadata>,
}

pub type PreviewResult = (u64, Result<Preview, String>);

pub fn start(result: impl Fn(PreviewResult) + Send + 'static) -> Sender<PreviewRequest> {
    let (send, receive) = channel();
    std::thread::spawn(move || run(receive, result));
    send
}

fn run(requests: Receiver<PreviewRequest>, result: impl Fn(PreviewResult)) {
    let mut cache = VecDeque::new();
    let mut pending = None;
    while let Some(first) = pending.take().or_else(|| requests.recv().ok()) {
        let request = requests.try_iter().last().unwrap_or(first);
        result((request.id, prepare(&request, &mut cache)));
        for path in request.neighbors {
            if let Ok(request) = requests.try_recv() {
                pending = Some(request);
                break;
            }
            if !cache.iter().any(|image| image.path == path)
                && let Ok(image) = CachedImage::load(&path)
            {
                // An idle prefetch uses only the space next to the active image.
                if cache_size(&cache) + image.stretched.len() <= MAX_CACHED_PIXELS {
                    cache.push_back(image);
                }
            }
        }
    }
}

const MAX_CACHED_PIXELS: usize = 256 * 1024 * 1024 / std::mem::size_of::<f64>();
const MAX_CACHE_ENTRIES: usize = 12;

fn cache_size(cache: &VecDeque<CachedImage>) -> usize {
    cache.iter().map(|entry| entry.stretched.len()).sum()
}

struct CachedImage {
    path: PathBuf,
    revision: Option<(u64, SystemTime)>,
    stretched: Vec<f64>,
    size: [usize; 2],
    metadata: Arc<Metadata>,
}

impl CachedImage {
    fn load(path: &Path) -> Result<Self, String> {
        let state = revision(path);
        let frame = load(path)?;
        if frame.width == 0 || frame.height == 0 || frame.pixels.len() != frame.width * frame.height
        {
            return Err("FITS preview contains inconsistent image dimensions".into());
        }
        Ok(Self {
            path: path.to_path_buf(),
            revision: state,
            stretched: statistical_stretch(&frame.pixels, StretchParams::default()),
            size: [frame.width, frame.height],
            metadata: Arc::new(frame.metadata),
        })
    }
}

fn revision(path: &Path) -> Option<(u64, SystemTime)> {
    let metadata = path.metadata().ok()?;
    Some((metadata.len(), metadata.modified().ok()?))
}

fn prepare(request: &PreviewRequest, cache: &mut VecDeque<CachedImage>) -> Result<Preview, String> {
    let cached = cache
        .iter()
        .position(|entry| entry.path == request.path)
        .and_then(|index| cache.remove(index));
    let current = revision(&request.path);
    let image = match cached {
        Some(image) if !request.refresh && current.is_some() && current == image.revision => image,
        _ => CachedImage::load(&request.path)?,
    };
    while !cache.is_empty() && cache_size(cache) + image.stretched.len() > MAX_CACHED_PIXELS {
        cache.pop_back();
    }
    let mut pixels = if request.automatic {
        render_auto_preview(&image.stretched)
    } else {
        render_preview(&image.stretched, request.adjustment)
    };
    let [width, height] = image.size;
    if width == 0 || height == 0 || pixels.len() != width * height {
        return Err("FITS preview produced an invalid grayscale image".into());
    }
    let scale = width.max(height).div_ceil(request.texture_limit.max(1));
    let texture_size = if scale > 1 {
        let mut reduced = Vec::with_capacity(width.div_ceil(scale) * height.div_ceil(scale));
        for row in pixels.chunks_exact(width).step_by(scale) {
            reduced.extend(row.iter().step_by(scale).copied());
        }
        pixels = reduced;
        let sampled_height = height.div_ceil(scale);
        let sampled_width = width.div_ceil(scale);
        [sampled_width, sampled_height]
    } else {
        image.size
    };
    if texture_size[0] == 0
        || texture_size[1] == 0
        || pixels.len() < texture_size[0] * texture_size[1]
    {
        return Err("FITS preview texture is smaller than its declared size".into());
    }
    let rendered = Preview {
        pixels,
        texture_size,
        image_size: image.size,
        metadata: image.metadata.clone(),
    };
    cache.push_front(image);
    while cache.len() > MAX_CACHE_ENTRIES {
        cache.pop_back();
    }
    Ok(rendered)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constant_frames_render_valid_preview_pixels() {
        let constant = statistical_stretch(&[42.0; 16], StretchParams::default());
        assert!(constant.iter().all(|pixel| pixel.is_finite()));
    }

    #[test]
    fn downsampled_texture_size_matches_pixel_buffer() {
        let request = PreviewRequest {
            id: 1,
            path: PathBuf::new(),
            adjustment: Adjustment::default(),
            automatic: true,
            refresh: false,
            neighbors: Vec::new(),
            texture_limit: 4,
        };
        let metadata = Arc::new(
            load(
                &Path::new(env!("CARGO_MANIFEST_DIR")).join("../frames/tests/reference/星点.fits"),
            )
            .unwrap()
            .metadata,
        );
        let rendered = prepare_bytes(&request, vec![0.5; 100], [10, 10], metadata).unwrap();
        assert_eq!(
            rendered.pixels.len(),
            rendered.texture_size[0] * rendered.texture_size[1]
        );
    }

    fn prepare_bytes(
        request: &PreviewRequest,
        stretched: Vec<f64>,
        size: [usize; 2],
        metadata: Arc<Metadata>,
    ) -> Result<Preview, String> {
        let mut pixels = if request.automatic {
            render_auto_preview(&stretched)
        } else {
            render_preview(&stretched, request.adjustment)
        };
        let [width, height] = size;
        let scale = width.max(height).div_ceil(request.texture_limit.max(1));
        let texture_size = if scale > 1 {
            let mut reduced = Vec::with_capacity(width.div_ceil(scale) * height.div_ceil(scale));
            for row in pixels.chunks_exact(width).step_by(scale) {
                reduced.extend(row.iter().step_by(scale).copied());
            }
            pixels = reduced;
            [width.div_ceil(scale), height.div_ceil(scale)]
        } else {
            size
        };
        Ok(Preview {
            pixels,
            texture_size,
            image_size: size,
            metadata,
        })
    }
}
