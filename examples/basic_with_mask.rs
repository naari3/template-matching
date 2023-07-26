use std::time::Instant;

use template_matching::{find_extremes, MatchTemplateMethod, TemplateMatcher};

fn main() {
    let input_image = image::load_from_memory(include_bytes!("ferris_with_background.png")).unwrap();
    let input_luma32f = input_image.to_luma32f();
    let template_image = image::load_from_memory(include_bytes!("ferris.png")).unwrap();
    let template_luma32f = template_image.to_luma32f();
    let mask_image = image::load_from_memory(include_bytes!("mask.png")).unwrap();
    let mask_luma32f = mask_image.to_luma32f();

    let mut matcher = TemplateMatcher::new();

    // Start matching with GPU acceleration
    let time = Instant::now();
    matcher.match_template_mask(
        &input_luma32f,
        &template_luma32f,
        &mask_luma32f,
        MatchTemplateMethod::SumOfSquaredDifferences,
    );
    let matcher_start_elapsed = time.elapsed();

    // Get result from GPU accelerated matching
    let time = Instant::now();
    let result = matcher.wait_for_result().unwrap();
    println!(
        "template_matching::match_template took {:.2} ms",
        (time.elapsed() + matcher_start_elapsed).as_micros() as f32 / 1000.0
    );

    let extremes = find_extremes(&result);
    println!("{:?}", extremes);
    println!();
}
