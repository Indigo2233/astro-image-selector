use std::collections::{BTreeSet, HashMap};
use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use astro_files::port::{FileManager, scan_cancellable};
use astro_frames::port::Adjustment;
use eframe::egui::{
    self, Align, Button, Color32, ColorImage, Context, FontData, FontDefinitions, FontFamily, Key,
    Label, Modifiers, ProgressBar, Rect, RichText, Sense, Stroke, TextureHandle, TextureOptions,
    Ui, Vec2,
};
use egui_extras::{Column, TableBuilder};

use crate::analysis::{AnalysisJob, Batch, auto_device, available_workers, default_workers};
use crate::catalog::{AnalysisResult, Metric, threshold_matches};
use crate::chart::{Chart, ChartState, filter_color};
use crate::preview::{self, Preview, PreviewRequest, PreviewResult};

const WINDOW_BACKGROUND: Color32 = Color32::from_gray(45);
const PANEL_BACKGROUND: Color32 = Color32::from_gray(58);
const INPUT_BACKGROUND: Color32 = Color32::from_gray(64);
const BORDER: Color32 = Color32::from_gray(85);
const MUTED: Color32 = Color32::from_gray(176);
const BLUE: Color32 = Color32::from_rgb(74, 144, 226);

pub fn launch(project: Option<PathBuf>) -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1370.0, 802.0])
            .with_min_inner_size([1170.0, 740.0]),
        renderer: eframe::Renderer::Glow,
        centered: true,
        ..Default::default()
    };
    eframe::run_native(
        "AstroImageSelector v1.1.0",
        options,
        Box::new(move |creation| Ok(Box::new(Desktop::new(creation.egui_ctx.clone(), project)))),
    )
}

fn receive_analysis_job(
    ctx: Context,
    send: Sender<Event>,
    id: u64,
    job: AnalysisJob,
) -> Option<JoinHandle<()>> {
    let worker_ctx = ctx.clone();
    let worker_send = send.clone();
    let handle = std::thread::Builder::new()
        .name("fits-analysis-result".into())
        .spawn(move || match job.join() {
            Ok(result) => signal(&worker_ctx, &worker_send, Event::Analyzed(id, Ok(result))),
            Err(error) => signal(&worker_ctx, &worker_send, Event::Analyzed(id, Err(error))),
        });
    match handle {
        Ok(handle) => Some(handle),
        Err(error) => {
            signal(
                &ctx,
                &send,
                Event::Analyzed(id, Err(format!("analysis result receiver failed: {error}"))),
            );
            None
        }
    }
}

enum Event {
    Scanned(u64, Option<Vec<PathBuf>>),
    AnalysisProgress(u64, usize, usize),
    Analyzed(u64, Result<Batch, String>),
    Moved(u64, Vec<PathBuf>, usize),
    Preview(PreviewResult),
}

struct Desktop {
    ctx: Context,
    send: Sender<Event>,
    receive: Receiver<Event>,
    preview_worker: Sender<PreviewRequest>,
    project: Option<PathBuf>,
    task_id: u64,
    analysis_cancel: Option<Arc<AtomicBool>>,
    analysis_receiver: Option<JoinHandle<()>>,
    cancel_scan: Option<Arc<AtomicBool>>,
    scanning: bool,
    moving_files: bool,
    progress: (usize, usize),
    workers: usize,
    use_gpu: bool,
    device_hint: String,
    files: Vec<PathBuf>,
    labels: Vec<String>,
    table_order: Vec<usize>,
    result_index: HashMap<PathBuf, usize>,
    results: Vec<AnalysisResult>,
    selected: Option<PathBuf>,
    rejected: BTreeSet<PathBuf>,
    scroll_to_file: bool,
    sort: Option<Metric>,
    metric: Metric,
    filter: Option<String>,
    threshold: f64,
    chart: ChartState,
    recenter_chart: bool,
    axes: [bool; 2],
    preview_id: u64,
    texture: Option<TextureHandle>,
    preview_size: [usize; 2],
    image_meta: String,
    preview_status: String,
    adjustment: Adjustment,
    adjustment_due: Option<Instant>,
    keep_view: bool,
    zoom: Option<f32>,
    fit_scale: f32,
    pan: Vec2,
    status: String,
    about_open: bool,
}

impl Desktop {
    fn new(ctx: Context, initial_project: Option<PathBuf>) -> Self {
        install_fonts(&ctx);
        install_style(&ctx);
        let preferred_device = auto_device();
        let (send, receive) = channel();
        let sender = send.clone();
        let preview_ctx = ctx.clone();
        let preview_worker = preview::start(move |result| {
            signal(&preview_ctx, &sender, Event::Preview(result));
        });
        let use_gpu = preferred_device.use_gpu();
        let device_hint = preferred_device.device_name().map_or_else(
            || "未检测到硬件计算设备，使用 CPU".into(),
            |name| {
                if use_gpu {
                    format!("自动选择: {name}")
                } else {
                    format!("检测到集成显卡: {name}，使用 CPU")
                }
            },
        );
        let mut app = Self {
            ctx,
            send,
            receive,
            preview_worker,
            project: None,
            task_id: 0,
            analysis_cancel: None,
            analysis_receiver: None,
            cancel_scan: None,
            scanning: false,
            moving_files: false,
            workers: default_workers(),
            use_gpu,
            device_hint,
            progress: (0, 0),
            files: Vec::new(),
            labels: Vec::new(),
            table_order: Vec::new(),
            result_index: HashMap::new(),
            results: Vec::new(),
            selected: None,
            rejected: BTreeSet::new(),
            scroll_to_file: false,
            sort: None,
            metric: Metric::default(),
            filter: None,
            threshold: Metric::default().limits().0,
            chart: ChartState::default(),
            recenter_chart: true,
            axes: [false, true],
            preview_id: 0,
            texture: None,
            preview_size: [0, 0],
            image_meta: String::new(),
            preview_status: "请选择 FITS 文件".into(),
            adjustment: Adjustment::default(),
            adjustment_due: None,
            keep_view: true,
            zoom: None,
            fit_scale: 1.0,
            pan: Vec2::ZERO,
            status: "就绪".into(),
            about_open: false,
        };
        if let Some(project) = initial_project {
            app.open_project(project);
        }
        app
    }

    fn choose_project(&mut self) {
        if self.moving_files {
            return;
        }
        let mut dialog = rfd::FileDialog::new().set_title("选择 FITS 图像目录");
        if let Some(project) = &self.project {
            dialog = dialog.set_directory(project);
        }
        if let Some(directory) = dialog.pick_folder() {
            self.open_project(directory);
        } else if !self.scanning && self.project.is_none() {
            self.status = "未选择目录".into();
        }
    }

    fn open_project(&mut self, directory: PathBuf) {
        self.stop_analysis();
        self.cancel_scan();
        self.task_id += 1;
        self.clear_preview();
        self.files.clear();
        self.rejected.clear();
        self.results.clear();
        self.rebuild_indexes();
        self.filter = None;
        self.metric = Metric::default();
        self.threshold = self.metric.limits().0;
        self.recenter_chart = true;
        self.scanning = true;
        self.status = "正在扫描 FITS 文件…".into();
        self.project = Some(directory.clone());
        let cancelled = Arc::new(AtomicBool::new(false));
        self.cancel_scan = Some(cancelled.clone());
        let id = self.task_id;
        let ctx = self.ctx.clone();
        let send = self.send.clone();
        std::thread::spawn(move || {
            let files = scan_cancellable(&directory, &cancelled);
            signal(
                &ctx,
                &send,
                Event::Scanned(id, (!cancelled.load(Ordering::Relaxed)).then_some(files)),
            );
        });
    }

    fn start_analysis(&mut self) {
        if self.files.is_empty()
            || self.scanning
            || self.moving_files
            || self.analysis_cancel.is_some()
        {
            return;
        }
        let Some(project) = self.project.clone() else {
            return;
        };
        self.task_id += 1;
        self.progress = (0, self.files.len());
        let id = self.task_id;
        let workers = self.workers;
        let use_gpu = self.use_gpu;
        let ctx = self.ctx.clone();
        let send = self.send.clone();
        let files = self.files.clone();
        match AnalysisJob::spawn(project, files, workers, use_gpu, move |current, total| {
            signal(&ctx, &send, Event::AnalysisProgress(id, current, total));
        }) {
            Ok(job) => {
                self.analysis_cancel = Some(job.cancel_token());
                self.analysis_receiver =
                    receive_analysis_job(self.ctx.clone(), self.send.clone(), id, job);
            }
            Err(error) => {
                self.progress = (0, self.files.len());
                self.status = format!("分析启动失败：{error}");
            }
        }
    }

    fn stop_analysis(&mut self) {
        if let Some(cancel) = self.analysis_cancel.take() {
            cancel.store(true, Ordering::Relaxed);
            self.task_id += 1;
            self.progress = (0, self.files.len());
            self.status = if self.files.is_empty() {
                "分析已停止".into()
            } else {
                format!("分析已停止，保留 {} 个待分析文件", self.files.len())
            };
        }
    }

    fn cancel_scan(&mut self) {
        if let Some(cancel) = self.cancel_scan.take() {
            cancel.store(true, Ordering::Relaxed);
            self.scanning = false;
        }
    }

    fn move_rejected(&mut self) {
        if self.rejected.is_empty() || self.moving_files || self.analysis_cancel.is_some() {
            return;
        }
        let Some(project) = self.project.clone() else {
            return;
        };
        self.task_id += 1;
        let id = self.task_id;
        let requested: Vec<_> = self.rejected.iter().cloned().collect();
        let count = requested.len();
        self.status = "正在移动标记文件…".into();
        self.moving_files = true;
        let ctx = self.ctx.clone();
        let send = self.send.clone();
        std::thread::spawn(move || {
            let manager = FileManager::new(project);
            let moved = requested
                .into_iter()
                .filter(|filename| manager.reject(filename).is_ok())
                .collect();
            signal(&ctx, &send, Event::Moved(id, moved, count));
        });
    }

    fn clear_preview(&mut self) {
        self.preview_id += 1;
        self.selected = None;
        self.texture = None;
        self.image_meta.clear();
        self.adjustment_due = None;
        self.preview_status = "请选择 FITS 文件".into();
        self.zoom = None;
        self.pan = Vec2::ZERO;
    }

    fn select(&mut self, file: PathBuf, scroll: bool) {
        if self.selected.as_ref() == Some(&file) {
            return;
        }
        self.selected = Some(file);
        self.scroll_to_file = scroll;
        self.texture = None;
        self.image_meta.clear();
        if !self.keep_view {
            self.zoom = None;
            self.pan = Vec2::ZERO;
        }
        self.request_preview(true, false);
    }

    fn request_preview(&mut self, automatic: bool, refresh: bool) {
        let Some(path) = self.selected.clone() else {
            return;
        };
        self.preview_id += 1;
        self.adjustment_due = None;
        self.preview_status = "正在处理预览…".into();
        let mut neighbors = Vec::new();
        if automatic
            && let Some(index) = self
                .table_order
                .iter()
                .position(|&index| self.files[index] == path)
        {
            for neighbor in [index.checked_sub(1), index.checked_add(1)]
                .into_iter()
                .flatten()
            {
                if let Some(&row) = self.table_order.get(neighbor) {
                    neighbors.push(self.files[row].clone());
                }
            }
        }
        let request = PreviewRequest {
            id: self.preview_id,
            path,
            adjustment: self.adjustment,
            automatic,
            refresh,
            neighbors,
            texture_limit: self.ctx.input(|input| input.max_texture_side),
        };
        if self.preview_worker.send(request).is_err() {
            self.preview_status = "预览线程已退出".into();
        }
    }

    fn apply_preview(&mut self, preview: Result<Preview, String>) {
        match preview {
            Ok(image) => {
                if image.pixels.len() != image.texture_size[0] * image.texture_size[1] {
                    self.texture = None;
                    self.image_meta.clear();
                    self.preview_status = "预览尺寸不一致，请重新选择图像".into();
                    return;
                }
                if image.pixels.iter().all(|pixel| *pixel == 0) {
                    self.texture = None;
                    self.preview_size = image.image_size;
                    self.image_meta = format!(
                        "{} × {} px  ·  {}  ·  {} s  ·  {}",
                        image.image_size[0],
                        image.image_size[1],
                        image.metadata.filter,
                        image.metadata.exposure,
                        image.metadata.object
                    );
                    self.preview_status = "预览没有可显示的有效亮度".into();
                    return;
                }
                self.texture = Some(self.ctx.load_texture(
                    "fits-preview",
                    ColorImage::from_gray(image.texture_size, &image.pixels),
                    TextureOptions {
                        magnification: egui::TextureFilter::Nearest,
                        ..TextureOptions::LINEAR
                    },
                ));
                self.preview_size = image.image_size;
                self.image_meta = format!(
                    "{} × {} px  ·  {}  ·  {} s  ·  {}",
                    self.preview_size[0],
                    self.preview_size[1],
                    image.metadata.filter,
                    image.metadata.exposure,
                    image.metadata.object
                );
                self.preview_status = "拉伸完成".into();
            }
            Err(error) => {
                self.texture = None;
                self.image_meta.clear();
                self.preview_status = format!("预览失败：{error}");
            }
        }
    }

    fn rebuild_indexes(&mut self) {
        if self.filter.as_ref().is_some_and(|filter| {
            self.results
                .iter()
                .all(|result| result.filter_type != *filter)
        }) {
            self.filter = None;
        }
        self.result_index = self
            .results
            .iter()
            .enumerate()
            .map(|(index, result)| (result.filename.clone(), index))
            .collect();
        let mut counts = HashMap::new();
        for file in &self.files {
            *counts.entry(file.file_name()).or_insert(0_usize) += 1;
        }
        self.labels = self
            .files
            .iter()
            .map(|file| {
                let name = file.file_name();
                if counts.get(&name).copied().unwrap_or(0) > 1
                    && let Some(relative) = self
                        .project
                        .as_ref()
                        .and_then(|root| file.strip_prefix(root).ok())
                {
                    return relative.display().to_string();
                }
                name.unwrap_or(file.as_os_str())
                    .to_string_lossy()
                    .into_owned()
            })
            .collect();
        self.resort();
    }

    fn resort(&mut self) {
        self.table_order = (0..self.files.len()).collect();
        let value = |index: usize, metric: Metric| {
            self.result_index
                .get(&self.files[index])
                .map(|&row| metric.value(&self.results[row]))
        };
        self.table_order.sort_by(|&a, &b| {
            if let Some(metric) = self.sort {
                let compare = match (value(a, metric), value(b, metric)) {
                    (Some(a), Some(b)) if metric.higher_is_better() => b.total_cmp(&a),
                    (Some(a), Some(b)) => a.total_cmp(&b),
                    (Some(_), None) => std::cmp::Ordering::Less,
                    (None, Some(_)) => std::cmp::Ordering::Greater,
                    (None, None) => std::cmp::Ordering::Equal,
                };
                if !compare.is_eq() {
                    return compare;
                }
            }
            self.labels[a]
                .to_lowercase()
                .cmp(&self.labels[b].to_lowercase())
        });
    }

    fn handle_events(&mut self) {
        while let Ok(event) = self.receive.try_recv() {
            match event {
                Event::Scanned(id, Some(files)) if id == self.task_id => {
                    self.cancel_scan = None;
                    self.files = files;
                    self.scanning = false;
                    self.rebuild_indexes();
                    self.status = if self.files.is_empty() {
                        "未找到 FITS 文件，请选择其他目录".into()
                    } else {
                        format!("已扫描 {} 个 FITS 文件", self.files.len())
                    };
                }
                Event::Scanned(id, None) if id == self.task_id => {
                    self.scanning = false;
                }
                Event::AnalysisProgress(id, current, total) if id == self.task_id => {
                    self.progress = (current, total);
                }
                Event::Analyzed(id, result) if id == self.task_id => {
                    self.analysis_cancel = None;
                    self.progress = (0, self.files.len());
                    match result {
                        Err(_) if self.status.starts_with("分析已停止") => {}
                        Ok(Batch {
                            results,
                            flat_count,
                            flat_files,
                            gpu,
                            gpu_count,
                            cache_hits,
                            cache_path,
                            cache_warning,
                        }) => {
                            self.files = results
                                .iter()
                                .map(|result| result.filename.clone())
                                .collect();
                            let excluded: BTreeSet<_> = flat_files.into_iter().collect();
                            self.files.retain(|file| !excluded.contains(file));
                            self.rejected.retain(|file| !excluded.contains(file));
                            let mut status = match gpu {
                                Some(device) if gpu_count > 0 => format!(
                                    "分析完成，共 {} 个文件。GPU {gpu_count} 张 · {device}",
                                    results.len()
                                ),
                                _ => format!("CPU 分析完成，共 {} 个文件", results.len()),
                            };
                            if cache_hits > 0 {
                                status.push_str(&format!(" · 缓存命中 {cache_hits} 张"));
                            }
                            if flat_count > 0 {
                                status.push_str(&format!(" · 已排除 {flat_count} 张平场"));
                            }
                            if let Some(path) = cache_path {
                                status.push_str(&format!(" · 已写入 {}", path.display()));
                            }
                            if let Some(warning) = cache_warning {
                                status.push_str(&format!(" · {warning}"));
                            }
                            self.status = status;
                            self.results = results;
                            self.rebuild_indexes();
                            self.set_average_threshold();
                            self.recenter_chart = true;
                        }
                        Err(error) => self.status = format!("分析失败：{error}"),
                    }
                }
                Event::Moved(id, moved, requested) if id == self.task_id => {
                    self.moving_files = false;
                    let moved: BTreeSet<_> = moved.into_iter().collect();
                    self.files.retain(|file| !moved.contains(file));
                    self.results
                        .retain(|result| !moved.contains(&result.filename));
                    self.rejected.retain(|file| !moved.contains(file));
                    if self
                        .selected
                        .as_ref()
                        .is_some_and(|file| moved.contains(file))
                    {
                        self.clear_preview();
                    }
                    self.rebuild_indexes();
                    let failures = requested - moved.len();
                    self.status = if failures == 0 {
                        format!("已移动 {} 个文件到 rejected", moved.len())
                    } else {
                        format!("已移动 {} 个文件，{failures} 个移动失败", moved.len())
                    };
                }
                Event::Preview((id, preview)) if id == self.preview_id => {
                    self.apply_preview(preview)
                }
                Event::Scanned(_, _) | Event::AnalysisProgress(_, _, _) | Event::Analyzed(_, _) => {
                }
                Event::Moved(_, _, _) | Event::Preview(_) => {}
            }
        }
    }

    fn set_average_threshold(&mut self) {
        self.threshold = self
            .metric
            .average_threshold(&self.results)
            .unwrap_or(self.metric.limits().0);
    }

    fn select_by_threshold(&mut self) {
        let mut count = 0;
        for filename in threshold_matches(
            &self.results,
            self.metric,
            self.threshold,
            self.filter.as_deref(),
        ) {
            count += 1;
            self.rejected.insert(filename.to_path_buf());
        }
        self.status = format!("阈值筛选标记 {count} 个文件");
    }

    fn menu(&mut self, ui: &mut Ui) {
        egui::MenuBar::new().ui(ui, |ui| {
            ui.menu_button("文件", |ui| {
                if ui.button("打开项目目录   Ctrl+O").clicked() {
                    ui.close();
                    self.choose_project();
                }
                if ui.button("退出   Ctrl+Q").clicked() {
                    ui.ctx().send_viewport_cmd(egui::ViewportCommand::Close);
                }
            });
            ui.menu_button("帮助", |ui| {
                if ui.button("关于").clicked() {
                    ui.close();
                    self.about_open = true;
                }
            });
        });
    }

    fn project_controls(&mut self, ui: &mut Ui) {
        let running = self.analysis_cancel.is_some();
        control_group(ui, "项目导入", |ui| {
            if let Some(project) = &self.project {
                let name = project
                    .file_name()
                    .unwrap_or(project.as_os_str())
                    .to_string_lossy();
                ui.add(Label::new(RichText::new(format!("目录: {name}")).strong()).truncate())
                    .on_hover_text(project.display().to_string());
            } else {
                ui.colored_label(MUTED, "未选择目录");
            }
            if action(ui, "选择FITS目录", !self.moving_files, ui.available_width()) {
                self.choose_project();
            }
            if self.scanning {
                ui.colored_label(MUTED, "正在扫描 FITS 文件…");
            } else if self.project.is_some() && self.files.is_empty() {
                ui.colored_label(MUTED, "当前目录没有 FITS 文件");
            }
        });
        control_group(ui, "分析控制", |ui| {
            let start_enabled =
                !running && !self.scanning && !self.moving_files && !self.files.is_empty();
            let start_label = if self.scanning {
                "扫描中…"
            } else if running {
                "分析中…"
            } else if self.files.is_empty() {
                "开始分析（无文件）"
            } else {
                "开始分析"
            };
            if action(ui, start_label, start_enabled, ui.available_width()) {
                self.start_analysis();
            }
            if action(ui, "停止分析", running, ui.available_width()) {
                self.stop_analysis();
            }
            ui.horizontal(|ui| {
                ui.label("线程数:");
                ui.with_layout(egui::Layout::right_to_left(Align::Center), |ui| {
                    if ui
                        .add_enabled(!running, Button::new("自动").small())
                        .clicked()
                    {
                        self.workers = default_workers();
                    }
                    ui.add_enabled(
                        !running,
                        egui::DragValue::new(&mut self.workers)
                            .range(1..=available_workers())
                            .speed(1.0),
                    );
                });
            });
            ui.add_enabled(
                !running,
                egui::Checkbox::new(&mut self.use_gpu, "GPU 快速模式"),
            );
            ui.add(Label::new(RichText::new(&self.device_hint).small().color(MUTED)).truncate())
                .on_hover_text(&self.device_hint);
            if running && !self.scanning {
                let (current, total) = self.progress;
                ui.add(
                    ProgressBar::new(current as f32 / total.max(1) as f32)
                        .fill(BLUE)
                        .text(format!("{current}/{total}")),
                );
            }
        });
        control_group(ui, "关键阈值设置", |ui| {
            ui.horizontal(|ui| {
                ui.label("指标:");
                ui.with_layout(egui::Layout::right_to_left(Align::Center), |ui| {
                    self.metric_selector(ui, "controls_metric");
                });
            });
            ui.horizontal(|ui| {
                ui.label("阈值:");
                ui.with_layout(egui::Layout::right_to_left(Align::Center), |ui| {
                    ui.add(
                        egui::DragValue::new(&mut self.threshold)
                            .range(0.0..=self.metric.limits().1)
                            .fixed_decimals(2)
                            .speed(0.01),
                    );
                });
            });
            let direction = if self.metric.higher_is_better() {
                "低于"
            } else {
                "高于"
            };
            ui.colored_label(MUTED, format!("{direction}此值的图像将被标记为不合格"));
            ui.horizontal(|ui| {
                let width = pair_width(ui);
                if action(ui, "自动阈值", true, width) {
                    self.threshold =
                        self.metric
                            .average_threshold(&self.results)
                            .unwrap_or(match self.metric {
                                Metric::Snr => 10.0,
                                Metric::Stars => 50.0,
                                Metric::Brightness => 1000.0,
                                metric => metric.limits().0,
                            });
                }
                if action(ui, "重置", true, width) {
                    self.set_average_threshold();
                }
            });
        });
        control_group(ui, "文件操作", |ui| {
            let move_enabled = !self.rejected.is_empty() && !running && !self.moving_files;
            if action(ui, "移动不合格文件", move_enabled, ui.available_width()) {
                self.move_rejected();
            }
            ui.colored_label(MUTED, "移动到项目的 rejected 文件夹");
            ui.colored_label(MUTED, format!("选中: {} 个文件", self.rejected.len()));
            let can_select = !self.results.is_empty() && !self.moving_files;
            if action(ui, "全选不合格", can_select, ui.available_width()) {
                self.select_by_threshold();
            }
            if action(ui, "清除选择", !self.moving_files, ui.available_width()) {
                self.rejected.clear();
            }
        });
    }
    fn metric_selector(&mut self, ui: &mut Ui, id: &str) {
        let before = self.metric;
        egui::ComboBox::from_id_salt(id)
            .selected_text(before.label())
            .show_ui(ui, |ui| {
                for metric in Metric::ALL {
                    ui.selectable_value(&mut self.metric, metric, metric.label());
                }
            });
        if self.metric != before {
            self.set_average_threshold();
            self.recenter_chart = true;
        }
    }

    fn threshold_slider(&mut self, ui: &mut Ui) {
        ui.label("阈值:");
        ui.spacing_mut().slider_width = (ui.available_width() - 168.0).max(80.0);
        ui.add(
            egui::Slider::new(&mut self.threshold, 0.0..=self.metric.limits().1).fixed_decimals(2),
        );
    }

    fn file_table(&mut self, ui: &mut Ui) {
        ui.horizontal(|ui| {
            ui.heading("文件列表");
            let previous = self.sort;
            ui.with_layout(egui::Layout::right_to_left(Align::Center), |ui| {
                egui::ComboBox::from_id_salt("file_sort")
                    .width(102.0)
                    .selected_text(previous.map_or("文件名", Metric::label))
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut self.sort, None, "文件名");
                        for metric in [
                            Metric::Eccentricity,
                            Metric::Fwhm,
                            Metric::Snr,
                            Metric::Stars,
                        ] {
                            ui.selectable_value(&mut self.sort, Some(metric), metric.label());
                        }
                    });
            });
            if previous != self.sort {
                self.resort();
            }
        });
        ui.horizontal(|ui| {
            let width = (ui.available_width() - 2.0 * ui.spacing().item_spacing.x) / 3.0;
            if action(ui, "全选", !self.moving_files, width) {
                self.rejected.extend(self.files.iter().cloned());
            }
            if action(ui, "反选", !self.moving_files, width) {
                self.rejected = self
                    .files
                    .iter()
                    .filter(|f| !self.rejected.contains(*f))
                    .cloned()
                    .collect();
            }
            if action(ui, "清除选择", !self.moving_files, width) {
                self.rejected.clear();
            }
        });
        let height = (ui.available_height() - 45.0).max(38.0);
        let filename_width = (ui.available_width() - 330.0).max(65.0);
        let mut table = TableBuilder::new(ui)
            .striped(true)
            .resizable(true)
            .min_scrolled_height(height)
            .max_scroll_height(height)
            .column(Column::exact(25.0))
            .column(Column::initial(filename_width).at_least(65.0).clip(true))
            .column(Column::initial(56.0).at_least(42.0).clip(true))
            .columns(Column::initial(45.0).at_least(37.0).clip(true), 4)
            .column(Column::initial(36.0).at_least(30.0).clip(true));
        if std::mem::take(&mut self.scroll_to_file)
            && let Some(selected) = &self.selected
            && let Some(index) = self
                .table_order
                .iter()
                .position(|&index| &self.files[index] == selected)
        {
            table = table.scroll_to_row(index, Some(Align::Center));
        }
        let mut selected = None;
        table
            .header(28.0, |mut row| {
                for label in [
                    "",
                    "文件名",
                    "滤镜",
                    "椭率",
                    "FWHM",
                    "SNR",
                    "星点数",
                    "排除",
                ] {
                    row.col(|ui| {
                        ui.strong(label);
                    });
                }
            })
            .body(|body| {
                body.rows(25.0, self.table_order.len(), |mut row| {
                    let row_index = row.index();
                    let index = self.table_order[row_index];
                    let path = &self.files[index];
                    row.set_selected(self.selected.as_ref() == Some(path));
                    let mut rejected = self.rejected.contains(path);
                    let previous = rejected;
                    row.col(|ui| {
                        ui.colored_label(MUTED, (row_index + 1).to_string());
                    });
                    row.col(|ui| {
                        let response = ui
                            .add(
                                Label::new(&self.labels[index])
                                    .truncate()
                                    .sense(Sense::click()),
                            )
                            .on_hover_text(path.display().to_string());
                        if response.clicked() {
                            selected = Some(path.clone());
                        }
                        response.context_menu(|ui| {
                            if ui
                                .add_enabled(
                                    !self.moving_files,
                                    Button::new(if rejected {
                                        "取消排除"
                                    } else {
                                        "标记为排除"
                                    }),
                                )
                                .clicked()
                            {
                                rejected = !rejected;
                                ui.close();
                            }
                            if ui.button("复制文件名").clicked() {
                                ui.ctx().copy_text(path.display().to_string());
                                ui.close();
                            }
                        });
                    });
                    let result = self
                        .result_index
                        .get(path)
                        .map(|&index| &self.results[index]);
                    row.col(|ui| {
                        if let Some(result) = result {
                            ui.colored_label(
                                filter_color(&result.filter_type),
                                &result.filter_type,
                            );
                        } else {
                            ui.label("—");
                        }
                    });
                    for metric in [
                        Metric::Eccentricity,
                        Metric::Fwhm,
                        Metric::Snr,
                        Metric::Stars,
                    ] {
                        row.col(|ui| {
                            let label = result.map_or_else(
                                || "—".into(),
                                |image| {
                                    if metric == Metric::Stars {
                                        image.star_count.to_string()
                                    } else {
                                        format!("{:.2}", metric.value(image))
                                    }
                                },
                            );
                            if ui.add(Label::new(label).sense(Sense::click())).clicked() {
                                selected = Some(path.clone());
                            }
                        });
                    }
                    row.col(|ui| {
                        ui.add_enabled(
                            !self.moving_files,
                            egui::Checkbox::without_text(&mut rejected),
                        );
                    });
                    if rejected && !previous {
                        self.rejected.insert(path.clone());
                    } else if !rejected && previous {
                        self.rejected.remove(path);
                    }
                });
            });
        if let Some(path) = selected {
            self.select(path, false);
        }
        let current = self
            .selected
            .as_ref()
            .and_then(|path| {
                self.table_order
                    .iter()
                    .position(|&row| &self.files[row] == path)
            })
            .map_or_else(|| "-".into(), |row| (row + 1).to_string());
        ui.colored_label(
            MUTED,
            format!(
                "总计: {}  当前: {current}  排除: {}",
                self.files.len(),
                self.rejected.len()
            ),
        );
    }

    fn quality_chart(&mut self, ui: &mut Ui) {
        ui.horizontal(|ui| {
            ui.heading("数据图表");
            ui.with_layout(egui::Layout::right_to_left(Align::Center), |ui| {
                let filters: BTreeSet<_> = self
                    .results
                    .iter()
                    .map(|image| image.filter_type.as_str())
                    .collect();
                egui::ComboBox::from_id_salt("chart_filter")
                    .width(76.0)
                    .selected_text(self.filter.as_deref().unwrap_or("全部"))
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut self.filter, None, "全部");
                        for filter in filters {
                            ui.selectable_value(&mut self.filter, Some(filter.to_string()), filter);
                        }
                    });
                ui.label("滤镜:");
                self.metric_selector(ui, "chart_metric");
                ui.label("指标:");
            });
        });
        ui.horizontal(|ui| {
            ui.label("缩放模式:");
            let mut only_x = self.axes == [true, false];
            let mut only_y = self.axes == [false, true];
            if ui.checkbox(&mut only_x, "仅X轴").changed() {
                self.axes = [true, !only_x];
            }
            if ui.checkbox(&mut only_y, "仅Y轴").changed() {
                self.axes = [!only_y, true];
            }
            ui.with_layout(egui::Layout::right_to_left(Align::Center), |ui| {
                if ui.button("居中序列").clicked() {
                    self.recenter_chart = true;
                }
            });
        });
        let chart_height = ui.available_height() - 58.0;
        if let Some(path) = self.chart.show(
            ui,
            Chart {
                results: &self.results,
                rejected: &self.rejected,
                selected: self.selected.as_deref(),
                filter: self.filter.as_deref(),
                metric: self.metric,
                threshold: &mut self.threshold,
                axis: self.axes,
                recenter: std::mem::take(&mut self.recenter_chart),
            },
            chart_height,
        ) {
            self.select(path, true);
        }
        ui.horizontal(|ui| {
            self.threshold_slider(ui);
            if action(
                ui,
                "全选不合格",
                !self.results.is_empty() && !self.moving_files,
                93.0,
            ) {
                self.select_by_threshold();
            }
        });
        self.chart_stats(ui);
    }
    fn chart_stats(&self, ui: &mut Ui) {
        let mut count = 0;
        let mut total = 0.0;
        let mut min = f64::INFINITY;
        let mut max = f64::NEG_INFINITY;
        for image in &self.results {
            if self
                .filter
                .as_ref()
                .is_some_and(|filter| image.filter_type != *filter)
            {
                continue;
            }
            let value = self.metric.plot_value(image);
            if value.is_finite() {
                count += 1;
                total += value;
                min = min.min(value);
                max = max.max(value);
            }
        }
        if count == 0 {
            ui.label("等待分析数据");
        } else {
            ui.label(format!(
                "{count} 项  ·  平均 {:.2}  ·  范围 {:.2}–{:.2}",
                total / f64::from(count),
                min,
                max
            ));
        }
    }

    fn stretch_controls(&mut self, ui: &mut Ui) {
        ui.heading("图像拉伸控制");
        control_group(ui, "高光保护", |ui| {
            if action(ui, "智能拉伸", true, ui.available_width()) {
                self.request_preview(true, true);
            }
        });
        control_group(ui, "手动微调", |ui| {
            let brightness =
                stretch_value(ui, "亮度:", &mut self.adjustment.brightness, -1.0..=1.0);
            let contrast = stretch_value(ui, "对比度:", &mut self.adjustment.contrast, 0.1..=3.0);
            let gamma = stretch_value(ui, "伽马值:", &mut self.adjustment.gamma, 0.3..=3.0);
            if (brightness || contrast || gamma) && self.selected.is_some() {
                let delay = Duration::from_millis(120);
                self.adjustment_due = Some(Instant::now() + delay);
                ui.ctx().request_repaint_after(delay);
            }
            if action(ui, "重置参数", true, ui.available_width()) {
                self.adjustment = Adjustment::default();
                self.request_preview(false, false);
            }
        });
        control_group(ui, "视图控制", |ui| {
            ui.horizontal(|ui| {
                let width = pair_width(ui);
                if action(ui, "放大", true, width) {
                    self.zoom_by(1.25, Vec2::ZERO);
                }
                if action(ui, "缩小", true, width) {
                    self.zoom_by(1.0 / 1.25, Vec2::ZERO);
                }
            });
            ui.horizontal(|ui| {
                let width = pair_width(ui);
                if action(ui, "适应窗口", true, width) {
                    self.zoom = None;
                    self.pan = Vec2::ZERO;
                }
                if action(ui, "重置视图", true, width) && self.texture.is_some() {
                    self.zoom = Some(1.0);
                    self.pan = Vec2::ZERO;
                }
            });
            ui.checkbox(&mut self.keep_view, "保持图像缩放（用于对比图像细节）");
        });
        control_group(ui, "状态信息", |ui| {
            ui.colored_label(
                MUTED,
                if self.selected.is_none() {
                    "就绪"
                } else {
                    &self.preview_status
                },
            );
        });
    }
    fn show_preview(&mut self, ui: &mut Ui) {
        let mut title = "图像预览".into();
        if let Some(file) = &self.selected {
            title = file
                .file_name()
                .unwrap_or(file.as_os_str())
                .to_string_lossy()
                .into_owned();
        }
        ui.horizontal(|ui| {
            ui.add(Label::new(RichText::new(title).strong()).truncate());
            ui.with_layout(egui::Layout::right_to_left(Align::Center), |ui| {
                let metadata = if self.image_meta.is_empty() {
                    "尺寸: - | 类型: -"
                } else {
                    &self.image_meta
                };
                ui.add(Label::new(RichText::new(metadata).color(MUTED)).truncate());
            });
        });
        let (area, response) = ui.allocate_exact_size(ui.available_size(), Sense::drag());
        let painter = ui.painter_at(area);
        painter.rect_filled(area, 0.0, Color32::BLACK);
        painter.rect_stroke(
            area,
            0.0,
            Stroke::new(1.0, BORDER),
            egui::StrokeKind::Inside,
        );
        let Some(texture) = self.texture.as_ref().map(TextureHandle::id) else {
            painter.text(
                area.center(),
                egui::Align2::CENTER_CENTER,
                &self.preview_status,
                egui::FontId::proportional(14.0),
                Color32::LIGHT_GRAY,
            );
            return;
        };
        let size = Vec2::new(self.preview_size[0] as f32, self.preview_size[1] as f32);
        if self.zoom.is_none() {
            self.fit_scale = (area.width() / size.x)
                .min(area.height() / size.y)
                .max(0.001);
            self.zoom = Some(self.fit_scale);
        }
        self.pan += response.drag_motion();
        if response.hovered() {
            let scroll: f32 = ui.input(|input| {
                input
                    .raw
                    .events
                    .iter()
                    .filter_map(|event| {
                        if let egui::Event::MouseWheel { delta, .. } = event {
                            Some(delta.y)
                        } else {
                            None
                        }
                    })
                    .sum()
            });
            if scroll != 0.0
                && let Some(mouse) = response.hover_pos()
            {
                let factor = if scroll > 0.0 { 1.25 } else { 1.0 / 1.25 };
                self.zoom_by(factor, mouse - area.center());
            }
        }
        let image =
            Rect::from_center_size(area.center() + self.pan, size * self.zoom.unwrap_or(1.0));
        let uv = Rect::from_min_max(egui::Pos2::ZERO, egui::pos2(1.0, 1.0));
        painter.image(texture, image, uv, Color32::WHITE);
    }

    fn zoom_by(&mut self, factor: f32, offset: Vec2) {
        if let Some(old) = self.zoom {
            let next = (old * factor).min(10.0).max(self.fit_scale.max(0.1));
            self.pan = offset + (self.pan - offset) * (next / old);
            self.zoom = Some(next);
        }
    }
}

impl eframe::App for Desktop {
    fn logic(&mut self, ctx: &Context, _frame: &mut eframe::Frame) {
        self.handle_events();
        if let Some(due) = self.adjustment_due {
            let remaining = due.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                self.request_preview(false, false);
            } else {
                ctx.request_repaint_after(remaining);
            }
        }
    }

    fn ui(&mut self, ui: &mut Ui, _frame: &mut eframe::Frame) {
        if ui.input_mut(|input| input.consume_key(Modifiers::CTRL, Key::O)) {
            self.choose_project();
        }
        if ui.input_mut(|input| input.consume_key(Modifiers::CTRL, Key::Q)) {
            ui.ctx().send_viewport_cmd(egui::ViewportCommand::Close);
        }
        egui::Panel::top("main_menu")
            .frame(
                egui::Frame::new()
                    .fill(PANEL_BACKGROUND)
                    .inner_margin(egui::Margin::symmetric(8, 3)),
            )
            .show(ui, |ui| self.menu(ui));
        egui::Panel::bottom("status")
            .frame(
                egui::Frame::new()
                    .fill(PANEL_BACKGROUND)
                    .inner_margin(egui::Margin::symmetric(8, 4)),
            )
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.label(&self.status);
                    ui.with_layout(egui::Layout::right_to_left(Align::Center), |ui| {
                        ui.colored_label(MUTED, format!("文件: {}", self.files.len()));
                    });
                });
            });
        egui::CentralPanel::default()
            .frame(
                egui::Frame::new()
                    .fill(WINDOW_BACKGROUND)
                    .inner_margin(egui::Margin::same(6)),
            )
            .show(ui, |ui| {
                let sidebar = (ui.available_width() * 0.20).clamp(275.0, 360.0);
                egui::Panel::left("project")
                    .resizable(true)
                    .default_size(sidebar)
                    .min_size(260.0)
                    .max_size(400.0)
                    .frame(pane_frame(12))
                    .show(ui, |ui| {
                        ui.set_min_height(ui.available_height());
                        egui::ScrollArea::vertical().show(ui, |ui| self.project_controls(ui));
                    });
                egui::CentralPanel::default()
                    .frame(grid_frame())
                    .show(ui, |ui| {
                        let top = (ui.available_height() * 0.38).max(250.0);
                        egui::Panel::top("table_and_chart")
                            .resizable(true)
                            .default_size(top)
                            .min_size(236.0)
                            .max_size((ui.available_height() - 280.0).max(236.0))
                            .frame(grid_frame())
                            .show(ui, |ui| {
                                ui.set_min_height(ui.available_height());
                                let table_width = ui.available_width() * 0.42;
                                egui::Panel::left("files_table")
                                    .resizable(true)
                                    .default_size(table_width)
                                    .min_size(353.0)
                                    .max_size((ui.available_width() - 340.0).max(353.0))
                                    .frame(pane_frame(8))
                                    .show(ui, |ui| self.file_table(ui));
                                egui::CentralPanel::default()
                                    .frame(pane_frame(8))
                                    .show(ui, |ui| self.quality_chart(ui));
                            });
                        egui::CentralPanel::default()
                            .frame(grid_frame())
                            .show(ui, |ui| {
                                let preview_width = ui.available_width() * 0.74;
                                egui::Panel::left("image_preview")
                                    .resizable(true)
                                    .default_size(preview_width)
                                    .min_size(350.0)
                                    .max_size((ui.available_width() - 244.0).max(350.0))
                                    .frame(pane_frame(8))
                                    .show(ui, |ui| self.show_preview(ui));
                                egui::CentralPanel::default().frame(pane_frame(10)).show(
                                    ui,
                                    |ui| {
                                        egui::ScrollArea::vertical()
                                            .show(ui, |ui| self.stretch_controls(ui));
                                    },
                                );
                            });
                    });
            });
        egui::Window::new("关于 AstroImageSelector")
            .collapsible(false)
            .open(&mut self.about_open)
            .show(ui.ctx(), |ui| {
                ui.heading("天文图像筛选软件");
                ui.label("FITS 图像质量分析、筛选和预览");
            });
    }
}

impl Drop for Desktop {
    fn drop(&mut self) {
        self.stop_analysis();
        self.cancel_scan();
    }
}

fn signal(ctx: &Context, sender: &Sender<Event>, message: Event) {
    if sender.send(message).is_ok() {
        ctx.request_repaint();
    }
}

fn install_fonts(ctx: &Context) {
    let windows = std::env::var_os("WINDIR").unwrap_or_else(|| OsStr::new("C:\\Windows").into());
    let folder = Path::new(&windows).join("Fonts");
    let Some(bytes) = ["msyh.ttc", "msjh.ttc", "simsun.ttc"]
        .iter()
        .find_map(|file| std::fs::read(folder.join(file)).ok())
    else {
        return;
    };
    let mut fonts = FontDefinitions::default();
    fonts
        .font_data
        .insert("cjk".into(), Arc::new(FontData::from_owned(bytes)));
    for family in [FontFamily::Proportional, FontFamily::Monospace] {
        fonts.families.entry(family).or_default().push("cjk".into());
    }
    ctx.set_fonts(fonts);
}

fn install_style(ctx: &Context) {
    let mut visuals = egui::Visuals::dark();
    visuals.panel_fill = PANEL_BACKGROUND;
    visuals.window_fill = PANEL_BACKGROUND;
    visuals.extreme_bg_color = Color32::BLACK;
    visuals.faint_bg_color = Color32::from_gray(51);
    visuals.selection.bg_fill = Color32::from_rgb(42, 90, 138);
    visuals.selection.stroke = Stroke::new(1.0, Color32::WHITE);
    visuals.widgets.noninteractive.bg_stroke = Stroke::new(1.0, BORDER);
    visuals.widgets.noninteractive.fg_stroke = Stroke::new(1.0, Color32::from_gray(224));
    visuals.slider_trailing_fill = true;
    for widget in [
        &mut visuals.widgets.inactive,
        &mut visuals.widgets.hovered,
        &mut visuals.widgets.active,
    ] {
        widget.bg_fill = INPUT_BACKGROUND;
        widget.weak_bg_fill = INPUT_BACKGROUND;
        widget.bg_stroke = Stroke::new(1.0, BORDER);
    }
    visuals.widgets.hovered.weak_bg_fill = Color32::from_gray(74);
    visuals.widgets.active.bg_stroke = Stroke::new(1.0, BLUE);
    ctx.set_theme(egui::Theme::Dark);
    ctx.set_visuals_of(egui::Theme::Dark, visuals);
    ctx.style_mut_of(egui::Theme::Dark, |style| {
        style.spacing.item_spacing = egui::vec2(6.0, 6.0);
        style.spacing.interact_size = egui::vec2(44.0, 24.0);
        style.spacing.button_padding = egui::vec2(9.0, 3.0);
        style
            .text_styles
            .insert(egui::TextStyle::Heading, egui::FontId::proportional(14.0));
        style
            .text_styles
            .insert(egui::TextStyle::Body, egui::FontId::proportional(12.0));
        style
            .text_styles
            .insert(egui::TextStyle::Button, egui::FontId::proportional(12.0));
    });
}

fn grid_frame() -> egui::Frame {
    egui::Frame::new().fill(WINDOW_BACKGROUND)
}

fn pane_frame(padding: i8) -> egui::Frame {
    egui::Frame::new()
        .fill(PANEL_BACKGROUND)
        .stroke(Stroke::new(1.0, BORDER))
        .corner_radius(6)
        .outer_margin(egui::Margin::same(3))
        .inner_margin(egui::Margin::same(padding))
}

fn control_group(ui: &mut Ui, title: &str, contents: impl FnOnce(&mut Ui)) {
    ui.add_space(8.0);
    let frame = egui::Frame::new()
        .fill(PANEL_BACKGROUND)
        .stroke(Stroke::new(1.0, BORDER))
        .corner_radius(8)
        .inner_margin(egui::Margin::same(9))
        .show(ui, |ui| {
            ui.set_min_width(ui.available_width());
            ui.add_space(5.0);
            contents(ui);
        });
    let origin = frame.response.rect.left_top() + egui::vec2(13.0, -8.0);
    let label = ui.painter().layout_no_wrap(
        title.to_owned(),
        egui::FontId::proportional(13.0),
        Color32::WHITE,
    );
    let rect = Rect::from_min_size(origin, label.size() + egui::vec2(16.0, 2.0));
    ui.painter().rect(
        rect,
        5.0,
        PANEL_BACKGROUND,
        Stroke::new(1.0, BORDER),
        egui::StrokeKind::Inside,
    );
    let mut caption = ui.new_child(
        egui::UiBuilder::new()
            .id_salt(title)
            .max_rect(rect.shrink2(egui::vec2(7.0, 0.0)))
            .layout(egui::Layout::left_to_right(Align::Center)),
    );
    caption.label(RichText::new(title).color(Color32::WHITE).strong());
    ui.add_space(4.0);
}

fn action(ui: &mut Ui, label: &str, enabled: bool, width: f32) -> bool {
    let fill = if enabled {
        BLUE
    } else {
        Color32::from_gray(102)
    };
    let size = egui::vec2(width, 32.0);
    ui.allocate_ui_with_layout(size, egui::Layout::left_to_right(Align::Center), |ui| {
        ui.set_max_width(width);
        ui.add_enabled(
            enabled,
            Button::new((
                egui::Atom::grow(),
                RichText::new(label).color(Color32::WHITE).strong(),
                egui::Atom::grow(),
            ))
            .fill(fill)
            .stroke(Stroke::NONE)
            .corner_radius(6)
            .min_size(size),
        )
        .clicked()
    })
    .inner
}

fn pair_width(ui: &Ui) -> f32 {
    (ui.available_width() - ui.spacing().item_spacing.x) * 0.5
}

fn stretch_value(
    ui: &mut Ui,
    label: &str,
    value: &mut f64,
    range: std::ops::RangeInclusive<f64>,
) -> bool {
    ui.horizontal(|ui| {
        ui.add_sized([43.0, 22.0], Label::new(label));
        ui.spacing_mut().slider_width = (ui.available_width() - 50.0).max(58.0);
        ui.add(egui::Slider::new(value, range).fixed_decimals(2))
            .changed()
    })
    .inner
}
