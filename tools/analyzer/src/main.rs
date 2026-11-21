use std::fs;
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::time::Instant;

use astro_files::port::scan;
use astro_selector::analysis::analyze_files;
use astro_selector::catalog::AnalysisStatus;

fn main() -> Result<(), String> {
    let mut args = std::env::args_os().skip(1);
    let source = PathBuf::from(args.next().ok_or("missing source directory")?);
    let result_directory = PathBuf::from(args.next().ok_or("missing result directory")?);
    let workers = args
        .next()
        .ok_or("missing worker count")?
        .to_string_lossy()
        .parse::<usize>()
        .map_err(|error| error.to_string())?;
    let use_gpu = args.next().is_some_and(|value| value == "gpu");
    fs::create_dir_all(&result_directory).map_err(|error| error.to_string())?;

    let files = scan(&source);
    let start = Instant::now();
    let batch = analyze_files(
        &result_directory,
        &source,
        &files,
        workers,
        use_gpu,
        &AtomicBool::new(false),
        |_, _| {},
    )?;
    let elapsed = start.elapsed();
    let statuses =
        batch
            .results
            .iter()
            .fold(std::collections::BTreeMap::new(), |mut counts, result| {
                *counts.entry(result.status.label()).or_insert(0) += 1;
                counts
            });
    let failures = batch
        .results
        .iter()
        .filter(|result| !matches!(result.status, AnalysisStatus::Success))
        .map(|result| {
            serde_json::json!({
                "path": result.filename,
                "status": result.status.label(),
                "error": result.error,
            })
        })
        .collect::<Vec<_>>();
    let report = serde_json::json!({
        "source": source,
        "workers": workers,
        "gpu": batch.gpu,
        "gpu_count": batch.gpu_count,
        "gpu_error": batch.gpu_error,
        "cache_hits": batch.cache_hits,
        "flat_count": batch.flat_count,
        "elapsed_ms": elapsed.as_millis(),
        "statuses": statuses,
        "failures": failures,
        "results": batch.results,
    });
    let destination = result_directory.join("benchmark.json");
    fs::write(
        &destination,
        serde_json::to_vec_pretty(&report).map_err(|error| error.to_string())?,
    )
    .map_err(|error| error.to_string())?;
    Ok(())
}
