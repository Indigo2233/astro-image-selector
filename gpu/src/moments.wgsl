struct CutoutShape {
    side: u32,
    area: u32,
    _padding: vec2<u32>,
}

@group(0) @binding(0) var<storage, read> pixels: array<f32>;
@group(0) @binding(1) var<storage, read> backgrounds: array<f32>;
@group(0) @binding(2) var<storage, read_write> metrics: array<vec4<f32>>;
@group(0) @binding(3) var<uniform> shape: CutoutShape;

const LANES: u32 = 64u;

var<workgroup> first_sums: array<vec3<f32>, LANES>;
var<workgroup> second_sums: array<vec2<f32>, LANES>;

fn position(pixel: u32) -> vec2<f32> {
    return vec2(f32(pixel % shape.side), f32(pixel / shape.side));
}

@compute @workgroup_size(LANES)
fn measure(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    let cutout = group.x;
    let offset = cutout * shape.area;
    let background = backgrounds[cutout];
    var first = vec3<f32>(0.0);
    for (var index = lane; index < shape.area; index += LANES) {
        let signal = max(pixels[offset + index] - background, 0.0);
        first += vec3(signal, signal * position(index));
    }
    first_sums[lane] = first;
    workgroupBarrier();
    for (var stride = LANES / 2u; stride > 0u; stride /= 2u) {
        if (lane < stride) {
            first_sums[lane] += first_sums[lane + stride];
        }
        workgroupBarrier();
    }

    let flux = first_sums[0].x;
    let safe_flux = select(1.0, flux, flux > 1e-6);
    let center = first_sums[0].yz / safe_flux;
    var second = vec2<f32>(0.0);
    for (var index = lane; index < shape.area; index += LANES) {
        let signal = max(pixels[offset + index] - background, 0.0);
        let displacement = position(index) - center;
        second += signal * (displacement * displacement);
    }
    second_sums[lane] = second;
    workgroupBarrier();
    for (var stride = LANES / 2u; stride > 0u; stride /= 2u) {
        if (lane < stride) {
            second_sums[lane] += second_sums[lane + stride];
        }
        workgroupBarrier();
    }

    if (lane == 0u) {
        let sigma = sqrt(max(second_sums[0] / safe_flux, vec2(1e-12)));
        let ratio = min(sigma.x, sigma.y) / max(max(sigma.x, sigma.y), 1e-6);
        let eccentricity = sqrt(clamp(1.0 - ratio * ratio, 0.0, 1.0));
        let fwhm = 2.3548 * sqrt(max(sigma.x * sigma.y, 0.0));
        metrics[cutout] = vec4(flux, fwhm, eccentricity, 0.0);
    }
}
