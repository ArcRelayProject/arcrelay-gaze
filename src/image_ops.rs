use image::{imageops::FilterType, DynamicImage, Rgb, RgbImage};
use imageproc::geometric_transformations::{rotate_about_center, Interpolation};

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

pub(crate) fn to_nchw_bgr(image: &RgbImage, width: u32, height: u32, eye_state: bool) -> Vec<f32> {
    let resized = DynamicImage::ImageRgb8(image.clone())
        .resize_exact(width, height, FilterType::Triangle)
        .to_rgb8();
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
