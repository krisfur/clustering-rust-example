use polars::prelude::*;
use linfa::prelude::*;
use linfa_clustering::KMeans;
use ndarray::{Array2, Array1};
use plotters::prelude::*;
use rand::thread_rng;
use rand_distr::{Normal, Distribution};
use std::fs::File;
use std::error::Error;

// generate a cluster with noise
fn generate_cluster(center: (f64, f64), std_dev: f64, n: usize) -> Vec<((f64, f64), (f64, f64))> {
    let mut rng = thread_rng();
    // create distributions centered on given point
    let dist_x = Normal::new(center.0, std_dev).unwrap();
    let dist_y = Normal::new(center.1, std_dev).unwrap();
    // generate points and noise based on it
    (0..n)
        .map(|_| {
            let x = dist_x.sample(&mut rng);
            let y = dist_y.sample(&mut rng);
            let noise_x = Normal::new(0.0, 0.05).unwrap().sample(&mut rng);
            let noise_y = Normal::new(0.0, 0.05).unwrap().sample(&mut rng);
            ((x, y), (x + noise_x, y + noise_y)) //outgoing vector of the form x0, y0, xn, yn
        })
        .collect()
}

fn main() -> Result<(), Box<dyn Error>> {
    // generate clustered data with noise baked in
    let mut points = Vec::new();
    points.extend(generate_cluster((2.0, 2.0), 0.3, 50));
    points.extend(generate_cluster((7.0, 7.0), 0.3, 50));
    points.extend(generate_cluster((2.0, 7.0), 0.3, 50));

    // separate into vectors
    let mut x = Vec::new();
    let mut y = Vec::new();
    let mut x_noisy = Vec::new();
    let mut y_noisy = Vec::new();

    for ((x0, y0), (xn, yn)) in points {
        x.push(x0);
        y.push(y0);
        x_noisy.push(xn);
        y_noisy.push(yn);
    }

    // create DataFrame -> keeping x,y without noise for inspecting results
    let mut df = df![
        "x" => &x,
        "y" => &y,
        "x_noisy" => &x_noisy,
        "y_noisy" => &y_noisy
    ]?;

    // convert to ndarray -> only using noisy data now
    let data: Array2<f64> = Array2::from_shape_vec(
        (x_noisy.len(), 2),
        x_noisy
            .iter()
            .zip(y_noisy.iter())
            .flat_map(|(&x, &y)| vec![x, y])
            .collect(),
    )
    .unwrap();

    // dummy targets (required by linfa but unused by KMeans)
    let dummy_targets: Array1<usize> = Array1::zeros(data.nrows());

    // create dataset and fit the model
    let dataset = DatasetBase::new(data.clone(), dummy_targets);
    let model = KMeans::params(3).fit(&dataset).unwrap();
    let preds = model.predict(&dataset);

    // add cluster column to the DataFrame
    let cluster_series = UInt32Chunked::from_iter(preds.iter().map(|&c| Some(c as u32)));
    df = df.hstack(&[Series::new("cluster", cluster_series)])?;

    // save the DataFrame to CSV
    CsvWriter::new(File::create("final_clustered.csv")?).finish(&mut df)?;

    // ==plotting time== //
    // base colours
    let base = RGBColor(30, 30, 46);
    let text = RGBColor(205, 214, 244);
    let grid = RGBColor(108, 112, 134);

    // cool colours that suit the darkmode background
    let cluster_colors = [
        RGBColor(250, 179, 135), // Peach
        RGBColor(203, 166, 247), // Mauve
        RGBColor(137, 220, 235), // Sky
    ];

    // set the canvas size
    let root = BitMapBackend::new("clusters.png", (1400, 800)).into_drawing_area();
    root.fill(&base)?; //apply dark background

    // build the chart itself
    let mut chart = ChartBuilder::on(&root)
        .margin(20)
        .caption("Noisy Clusters", ("sans-serif", 42).into_font().color(&text)) //set title text and font size here
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(0.5f64..8.5f64, 0.5f64..8.5f64)?; //set x and y limits here

    chart
        .configure_mesh()
        .axis_style(&text)
        .light_line_style(&grid)
        .label_style(("sans-serif", 32).into_font().color(&text)) //set axis font size here
        .draw()?;

    // draw points with cluster-specific colors
    for i in 0..df.height() {
        let x = df.column("x_noisy")?.get(i)?.try_extract::<f64>()?;
        let y = df.column("y_noisy")?.get(i)?.try_extract::<f64>()?;
        let cluster = df.column("cluster")?.get(i)?.try_extract::<u32>()?;

        let color = cluster_colors
            .get(cluster as usize)
            .unwrap_or(&RGBColor(255, 255, 255)); //fallback white

        chart.draw_series(std::iter::once(Circle::new((x, y), 4, color.filled())))?;
    }

    println!("✅ Saved clustered DataFrame and plot.");
    Ok(())
}