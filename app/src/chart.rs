use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use eframe::egui::{self, Color32, PointerButton, Ui};
use egui_plot::{
    HLine, HoverPosition, LineStyle, Plot, PlotBounds, PlotPoint, PlotResponse, Points,
};

use astro_selector::catalog::{AnalysisResult, Metric};

pub struct Chart<'a> {
    pub results: &'a [AnalysisResult],
    pub rejected: &'a BTreeSet<PathBuf>,
    pub selected: Option<&'a Path>,
    pub filter: Option<&'a str>,
    pub metric: Metric,
    pub threshold: &'a mut f64,
    pub axis: [bool; 2],
    pub recenter: bool,
}

#[derive(Default)]
pub struct ChartState {
    moving_threshold: bool,
}

impl ChartState {
    pub fn show(&mut self, ui: &mut Ui, chart: Chart<'_>, height: f32) -> Option<PathBuf> {
        let mut groups: BTreeMap<(&str, bool), Vec<[f64; 2]>> = BTreeMap::new();
        let mut candidates = Vec::new();
        let mut selected = None;
        for (index, image) in chart.results.iter().enumerate() {
            let value = chart.metric.plot_value(image);
            if !value.is_finite()
                || chart
                    .filter
                    .is_some_and(|filter| image.filter_type != filter)
            {
                continue;
            }
            let x = index as f64;
            let point = [x, value];
            let rejected = chart.rejected.contains(&image.filename);
            groups
                .entry((image.filter_type.as_str(), rejected))
                .or_default()
                .push(point);
            candidates.push((PlotPoint::new(x, value), image.filename.as_path()));
            if Some(image.filename.as_path()) == chart.selected {
                selected = Some(point);
            }
        }

        let results = chart.results;
        let metric = chart.metric;
        let zoom_label = match chart.axis {
            [true, false] => " (仅X轴缩放)",
            [false, true] => " (仅Y轴缩放)",
            _ => "",
        };
        let mut plot = Plot::new("quality-metrics")
            .height(height.max(102.0))
            .x_axis_label("文件序号")
            .y_axis_label(format!("{}{zoom_label}", metric.label()))
            .allow_zoom(chart.axis)
            .allow_drag(!self.moving_threshold)
            .allow_scroll(true)
            .allow_boxed_zoom(true)
            .show_x(false)
            .show_y(false)
            .include_y(0.0)
            .include_y(*chart.threshold)
            .label_formatter(move |hover| {
                let HoverPosition::NearDataPoint { position, .. } = hover else {
                    return None;
                };
                let result = results.get(position.x.round() as usize)?;
                let name = result.filename.file_name()?.to_string_lossy();
                Some(format!(
                    "{name}\n滤镜: {}\n{}: {:.3}",
                    result.filter_type,
                    metric.label(),
                    position.y
                ))
            });
        if chart.recenter {
            plot = plot.reset();
        }
        let threshold = *chart.threshold;
        let response = plot.show(ui, |plot_ui| {
            if chart.recenter && results.is_empty() {
                plot_ui.set_plot_bounds(PlotBounds::from_min_max(
                    [0.0, 0.0],
                    [100.0, metric.limits().1],
                ));
            }
            for ((filter, rejected), values) in groups {
                let color = if rejected {
                    Color32::from_rgb(244, 98, 88)
                } else {
                    filter_color(filter)
                };
                plot_ui.points(Points::new(filter, values).radius(3.8).color(color));
            }
            if let Some(point) = selected {
                plot_ui.points(
                    Points::new("当前文件", vec![point])
                        .radius(6.0)
                        .color(Color32::WHITE),
                );
            }
            plot_ui.hline(
                HLine::new("阈值", threshold)
                    .color(Color32::from_rgb(239, 70, 70))
                    .style(LineStyle::dashed_dense())
                    .width(1.5),
            );
        });

        {
            let frame = *response.transform.frame();
            let y = response
                .transform
                .position_from_point(&PlotPoint::new(0.0, threshold))
                .y;
            let clipped = y.clamp(frame.top(), frame.bottom());
            let (top, bottom) = if chart.metric.higher_is_better() {
                (clipped, frame.bottom())
            } else {
                (frame.top(), clipped)
            };
            let area = egui::Rect::from_min_max(
                egui::pos2(frame.left(), top),
                egui::pos2(frame.right(), bottom),
            );
            let painter = ui.painter_at(frame);
            painter.rect_filled(area, 0.0, Color32::from_rgba_unmultiplied(210, 31, 31, 36));
            if (frame.top() + 14.0..=frame.bottom()).contains(&y) {
                painter.text(
                    egui::pos2(frame.center().x, y - 3.0),
                    egui::Align2::CENTER_BOTTOM,
                    format!("阈值={threshold:.2}"),
                    egui::FontId::proportional(11.0),
                    Color32::LIGHT_GRAY,
                );
            }
        }

        self.interact(ui, &response, chart, &candidates)
    }

    fn interact(
        &mut self,
        ui: &Ui,
        response: &PlotResponse<()>,
        chart: Chart<'_>,
        candidates: &[(PlotPoint, &Path)],
    ) -> Option<PathBuf> {
        let pointer = ui.input(|input| input.pointer.interact_pos());
        if response.response.drag_started_by(PointerButton::Primary)
            && let Some(origin) = ui.input(|input| input.pointer.press_origin())
        {
            let line_y = response
                .transform
                .position_from_point(&PlotPoint::new(0.0, *chart.threshold))
                .y;
            self.moving_threshold = (origin.y - line_y).abs() <= 7.0;
        }
        if self.moving_threshold {
            if let Some(pointer) = pointer {
                let new_limit = response.transform.value_from_position(pointer).y;
                *chart.threshold = new_limit.clamp(0.0, chart.metric.limits().1);
            }
            self.moving_threshold = ui.input(|i| i.pointer.primary_down());
            return None;
        }
        let mouse = pointer.filter(|_| response.response.clicked())?;
        candidates
            .iter()
            .map(|(point, path)| {
                let distance = response
                    .transform
                    .position_from_point(point)
                    .distance_sq(mouse);
                (distance, *path)
            })
            .filter(|(distance, _)| *distance <= 144.0)
            .min_by(|a, b| a.0.total_cmp(&b.0))
            .map(|(_, path)| path.to_path_buf())
    }
}

pub fn filter_color(filter: &str) -> egui::Color32 {
    match filter {
        "L" => Color32::from_rgb(220, 226, 235),
        "R" => Color32::from_rgb(232, 102, 115),
        "G" => Color32::from_rgb(104, 197, 127),
        "B" => Color32::from_rgb(103, 156, 246),
        "S" | "SII" => Color32::from_rgb(239, 163, 98),
        "H" | "Ha" => Color32::from_rgb(216, 124, 225),
        "O" | "OIII" => Color32::from_rgb(109, 213, 219),
        _ => Color32::from_rgb(188, 177, 139),
    }
}
