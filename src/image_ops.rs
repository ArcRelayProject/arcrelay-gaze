use image::{imageops::FilterType, Rgb, RgbImage};
use imageproc::geometric_transformations::{
    rotate_about_center, warp_into, Interpolation, Projection,
};

use crate::{Point, Rect};

pub(crate) fn crop(image: &RgbImage, rect: Rect) -> Option<RgbImage> {
    let rect = rect.clamp(image.width(), image.height())?;
    Some(
        image::imageops::crop_imm(
            image,
            rect.x.round() as u32,
            rect.y.round() as u32,
            rect.width.round().max(1.0) as u32,
            rect.height.round().max(1.0) as u32,
        )
        .to_image(),
    )
}

pub(crate) fn eye_box(a: Point, b: Point, scale: f32, image: &RgbImage) -> Option<Rect> {
    let midpoint = a.midpoint(b);
    let size = a.distance(b) * scale;
    Rect {
        x: midpoint.x - size * 0.5,
        y: midpoint.y - size * 0.5,
        width: size,
        height: size,
    }
    .clamp(image.width(), image.height())
}

pub(crate) fn rotate_rgb(image: &RgbImage, degrees: f32) -> RgbImage {
    rotate_about_center(
        image,
        degrees.to_radians(),
        Interpolation::Bilinear,
        Rgb([0, 0, 0]),
    )
}

/// Align a face to the five-point geometry expected by Intel's
/// face-reidentification-retail-0095 model. The 35-point landmark model stores
/// eye corners at 0..=3, the nose tip at 4, and outer lip corners at 8 and 9.
pub(crate) fn align_face_for_embedding(image: &RgbImage, landmarks: &[Point]) -> Option<RgbImage> {
    const SIZE: f32 = 128.0;
    const TARGETS: [(f32, f32); 5] = [
        (0.315_568_75 * SIZE, 0.461_574_1 * SIZE),
        (0.682_622_9 * SIZE, 0.461_574_1 * SIZE),
        (0.500_262_5 * SIZE, 0.640_505_4 * SIZE),
        (0.349_471_87 * SIZE, 0.824_691_95 * SIZE),
        (0.653_436_5 * SIZE, 0.824_691_95 * SIZE),
    ];
    if landmarks.len() < 10 {
        return None;
    }
    let sources = [
        landmarks[0].midpoint(landmarks[1]),
        landmarks[2].midpoint(landmarks[3]),
        landmarks[4],
        landmarks[8],
        landmarks[9],
    ];
    let projection = similarity_projection(&sources, &TARGETS)?;
    let mut output = RgbImage::new(SIZE as u32, SIZE as u32);
    warp_into(
        image,
        &projection,
        Interpolation::Bilinear,
        Rgb([0, 0, 0]),
        &mut output,
    );
    Some(output)
}

fn similarity_projection(sources: &[Point; 5], targets: &[(f32, f32); 5]) -> Option<Projection> {
    let source_center = sources.iter().fold(Point::default(), |mut total, point| {
        total.x += point.x / sources.len() as f32;
        total.y += point.y / sources.len() as f32;
        total
    });
    let target_center = targets.iter().fold((0.0, 0.0), |mut total, point| {
        total.0 += point.0 / targets.len() as f32;
        total.1 += point.1 / targets.len() as f32;
        total
    });
    let mut denominator = 0.0;
    let mut a_numerator = 0.0;
    let mut b_numerator = 0.0;
    for (source, target) in sources.iter().zip(targets) {
        let x = source.x - source_center.x;
        let y = source.y - source_center.y;
        let u = target.0 - target_center.0;
        let v = target.1 - target_center.1;
        denominator += x * x + y * y;
        a_numerator += x * u + y * v;
        b_numerator += x * v - y * u;
    }
    if !denominator.is_finite() || denominator <= f32::EPSILON {
        return None;
    }
    let a = a_numerator / denominator;
    let b = b_numerator / denominator;
    let tx = target_center.0 - a * source_center.x + b * source_center.y;
    let ty = target_center.1 - b * source_center.x - a * source_center.y;
    Projection::from_matrix([a, -b, tx, b, a, ty, 0.0, 0.0, 1.0])
}

pub(crate) fn to_nchw_bgr(image: &RgbImage, width: u32, height: u32, eye_state: bool) -> Vec<f32> {
    // `DynamicImage::ImageRgb8(image.clone()).resize_exact(...)` used to copy
    // the complete source before every model stage. A normal gaze frame enters
    // this function eight times, so that hidden clone was a material part of
    // the continuous CPU and allocation cost. `imageops::resize` borrows the
    // source and allocates only the model-sized result.
    let resized = image::imageops::resize(image, width, height, FilterType::Triangle);
    let plane = (width * height) as usize;
    let mut output = vec![0.0; plane * 3];
    for (index, pixel) in resized.pixels().enumerate() {
        for (channel, value) in [pixel[2], pixel[1], pixel[0]].into_iter().enumerate() {
            let value = f32::from(value);
            output[channel * plane + index] = if eye_state {
                (value - 127.0) / 255.0
            } else {
                value
            };
        }
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn similarity_projection_maps_all_reference_points() {
        let sources = [
            Point { x: 10.0, y: 20.0 },
            Point { x: 30.0, y: 20.0 },
            Point { x: 20.0, y: 30.0 },
            Point { x: 12.0, y: 40.0 },
            Point { x: 28.0, y: 40.0 },
        ];
        let targets = [
            (30.0, 35.0),
            (70.0, 35.0),
            (50.0, 55.0),
            (34.0, 75.0),
            (66.0, 75.0),
        ];
        let projection = similarity_projection(&sources, &targets).unwrap();
        for (source, target) in sources.iter().zip(targets) {
            let actual = projection * (source.x, source.y);
            assert!((actual.0 - target.0).abs() < 1e-4);
            assert!((actual.1 - target.1).abs() < 1e-4);
        }
    }
}
