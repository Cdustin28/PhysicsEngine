//! Minimal SVG line and scatter charts, so the page needs no JavaScript.

use std::fmt::Write as _;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Each chart's clip path needs an id unique on the page.
static NEXT_ID: AtomicUsize = AtomicUsize::new(0);

/// Iowa State cardinal and gold, plus two neutrals for extra series.
pub const CARDINAL: &str = "#C8102E";
pub const GOLD: &str = "#F1BE48";
pub const SLATE: &str = "#3C3C3C";
pub const BLUE: &str = "#1F4E9A";

#[derive(Clone, Copy, PartialEq)]
pub enum Style {
    /// A line (theory).
    Line,
    /// Markers (measurements).
    Points,
    /// Hollow markers, for a second measurement that may sit on the first.
    Rings,
}

pub struct Series {
    pub name: String,
    pub points: Vec<(f64, f64)>,
    pub style: Style,
    pub color: &'static str,
}

impl Series {
    pub fn line(name: &str, points: Vec<(f64, f64)>, color: &'static str) -> Self {
        Self {
            name: name.into(),
            points,
            style: Style::Line,
            color,
        }
    }

    pub fn points(name: &str, points: Vec<(f64, f64)>, color: &'static str) -> Self {
        Self {
            name: name.into(),
            points,
            style: Style::Points,
            color,
        }
    }
}

impl Series {
    pub fn rings(name: &str, points: Vec<(f64, f64)>, color: &'static str) -> Self {
        Self {
            name: name.into(),
            points,
            style: Style::Rings,
            color,
        }
    }
}

pub struct Chart {
    pub x_label: String,
    pub y_label: String,
    pub series: Vec<Series>,
    pub log_x: bool,
    pub log_y: bool,
    /// Force the y range to include zero.
    pub y_from_zero: bool,
}

impl Chart {
    pub fn new(x_label: &str, y_label: &str) -> Self {
        Self {
            x_label: x_label.into(),
            y_label: y_label.into(),
            series: Vec::new(),
            log_x: false,
            log_y: false,
            y_from_zero: false,
        }
    }

    pub fn with(mut self, s: Series) -> Self {
        self.series.push(s);
        self
    }

    pub fn log_log(mut self) -> Self {
        self.log_x = true;
        self.log_y = true;
        self
    }

    pub fn include_zero(mut self) -> Self {
        self.y_from_zero = true;
        self
    }

    pub fn svg(&self) -> String {
        const W: f64 = 640.0;
        const L: f64 = 64.0;
        const R: f64 = 16.0;
        const B: f64 = 48.0;
        const PLOT_H: f64 = 296.0;
        // The legend sits above the plot, wrapping into rows; series with an
        // empty name are left out of it.
        let mut legend = Vec::new();
        let (mut lx, mut row) = (L, 0usize);
        for s in self.series.iter().filter(|s| !s.name.is_empty()) {
            let width = 24.0 + 6.6 * s.name.chars().count() as f64 + 18.0;
            if lx > L && lx + width > W - R {
                lx = L;
                row += 1;
            }
            legend.push((s, lx, row));
            lx += width;
        }
        let rows = if legend.is_empty() { 0 } else { row + 1 };
        let top = 12.0 + 18.0 * rows as f64;
        let height = top + PLOT_H + B;
        let tx = |v: f64| if self.log_x { v.log10() } else { v };
        let ty = |v: f64| if self.log_y { v.log10() } else { v };
        let all: Vec<(f64, f64)> = self
            .series
            .iter()
            .flat_map(|s| s.points.iter().map(|&(x, y)| (tx(x), ty(y))))
            .filter(|(x, y)| x.is_finite() && y.is_finite())
            .collect();
        let (mut x0, mut x1) = bounds(all.iter().map(|p| p.0));
        let (mut y0, mut y1) = bounds(all.iter().map(|p| p.1));
        if self.y_from_zero && !self.log_y {
            y0 = y0.min(0.0);
            y1 = y1.max(0.0);
        }
        let pad = |a: &mut f64, b: &mut f64, f: f64| {
            let span = (*b - *a).max(1e-12);
            *a -= span * f;
            *b += span * f;
        };
        pad(&mut x0, &mut x1, 0.02);
        pad(&mut y0, &mut y1, 0.06);
        let sx = |x: f64| L + (x - x0) / (x1 - x0) * (W - L - R);
        let sy = |y: f64| height - B - (y - y0) / (y1 - y0) * (height - top - B);

        let mut svg = String::new();
        let _ = write!(
            svg,
            r#"<svg viewBox="0 0 {W} {height}" role="img" xmlns="http://www.w3.org/2000/svg" font-family="inherit" font-size="12">"#
        );
        // Grid and ticks.
        for (v, label) in ticks(x0, x1, self.log_x) {
            let x = sx(v);
            let _ = write!(
                svg,
                r##"<line x1="{x:.1}" y1="{top}" x2="{x:.1}" y2="{:.1}" stroke="#e4e4e4"/><text x="{x:.1}" y="{:.1}" text-anchor="middle" fill="#555">{label}</text>"##,
                height - B,
                height - B + 16.0
            );
        }
        for (v, label) in ticks(y0, y1, self.log_y) {
            let y = sy(v);
            let _ = write!(
                svg,
                r##"<line x1="{L}" y1="{y:.1}" x2="{:.1}" y2="{y:.1}" stroke="#e4e4e4"/><text x="{:.1}" y="{:.1}" text-anchor="end" fill="#555">{label}</text>"##,
                W - R,
                L - 6.0,
                y + 4.0
            );
        }
        let _ = write!(
            svg,
            r##"<rect x="{L}" y="{top}" width="{:.1}" height="{:.1}" fill="none" stroke="#999"/>"##,
            W - L - R,
            height - top - B
        );
        let _ = write!(
            svg,
            r##"<text x="{:.1}" y="{:.1}" text-anchor="middle" fill="#333">{}</text>"##,
            L + (W - L - R) / 2.0,
            height - 10.0,
            escape(&self.x_label)
        );
        let _ = write!(
            svg,
            r##"<text transform="translate(16,{:.1}) rotate(-90)" text-anchor="middle" fill="#333">{}</text>"##,
            top + (height - top - B) / 2.0,
            escape(&self.y_label)
        );
        // Data, clipped to the plot area.
        let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
        let _ = write!(
            svg,
            r#"<clipPath id="clip{id}"><rect x="{L}" y="{top}" width="{:.1}" height="{:.1}"/></clipPath><g clip-path="url(#clip{id})">"#,
            W - L - R,
            height - top - B
        );
        for s in &self.series {
            let pts: Vec<(f64, f64)> = s
                .points
                .iter()
                .map(|&(x, y)| (tx(x), ty(y)))
                .filter(|(x, y)| x.is_finite() && y.is_finite())
                .map(|(x, y)| (sx(x), sy(y)))
                .collect();
            match s.style {
                Style::Line => {
                    let d: Vec<String> =
                        pts.iter().map(|(x, y)| format!("{x:.1},{y:.1}")).collect();
                    let _ = write!(
                        svg,
                        r#"<polyline points="{}" fill="none" stroke="{}" stroke-width="2"/>"#,
                        d.join(" "),
                        s.color
                    );
                }
                Style::Points => {
                    for (x, y) in pts {
                        let _ = write!(
                            svg,
                            r##"<circle cx="{x:.1}" cy="{y:.1}" r="3.5" fill="{}" stroke="#fff" stroke-width="0.8"/>"##,
                            s.color
                        );
                    }
                }
                Style::Rings => {
                    for (x, y) in pts {
                        let _ = write!(
                            svg,
                            r#"<circle cx="{x:.1}" cy="{y:.1}" r="6" fill="none" stroke="{}" stroke-width="1.8"/>"#,
                            s.color
                        );
                    }
                }
            }
        }
        svg.push_str("</g>");
        // Legend.
        for (s, lx, row) in legend {
            let ly = 14.0 + 18.0 * row as f64;
            match s.style {
                Style::Line => {
                    let _ = write!(
                        svg,
                        r#"<line x1="{lx:.1}" y1="{:.1}" x2="{:.1}" y2="{:.1}" stroke="{}" stroke-width="2"/>"#,
                        ly - 4.0,
                        lx + 18.0,
                        ly - 4.0,
                        s.color
                    );
                }
                Style::Points => {
                    let _ = write!(
                        svg,
                        r#"<circle cx="{:.1}" cy="{:.1}" r="3.5" fill="{}"/>"#,
                        lx + 9.0,
                        ly - 4.0,
                        s.color
                    );
                }
                Style::Rings => {
                    let _ = write!(
                        svg,
                        r#"<circle cx="{:.1}" cy="{:.1}" r="5" fill="none" stroke="{}" stroke-width="1.8"/>"#,
                        lx + 9.0,
                        ly - 4.0,
                        s.color
                    );
                }
            }
            let _ = write!(
                svg,
                r##"<text x="{:.1}" y="{ly:.1}" fill="#333">{}</text>"##,
                lx + 24.0,
                escape(&s.name)
            );
        }
        svg.push_str("</svg>");
        svg
    }
}

fn bounds(values: impl Iterator<Item = f64>) -> (f64, f64) {
    let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
    for v in values {
        lo = lo.min(v);
        hi = hi.max(v);
    }
    if !lo.is_finite() {
        return (0.0, 1.0);
    }
    if hi - lo < 1e-12 {
        return (lo - 0.5, hi + 0.5);
    }
    (lo, hi)
}

/// About five "nice" tick values (1, 2, 5 × 10ⁿ) in [lo, hi], with labels.
/// In log mode the axis values are log10 and ticks fall on powers of ten.
fn ticks(lo: f64, hi: f64, log: bool) -> Vec<(f64, String)> {
    if log {
        return (lo.ceil() as i32..=hi.floor() as i32)
            .map(|e| (f64::from(e), format_number(10f64.powi(e))))
            .collect();
    }
    let raw = (hi - lo) / 5.0;
    let mag = 10f64.powf(raw.log10().floor());
    let step = [1.0, 2.0, 5.0, 10.0]
        .iter()
        .map(|m| m * mag)
        .find(|s| *s >= raw)
        .unwrap_or(10.0 * mag);
    let mut out = Vec::new();
    let mut v = (lo / step).ceil() * step;
    while v <= hi + 1e-9 * step {
        out.push((
            v,
            format_number(if v.abs() < 1e-12 * step { 0.0 } else { v }),
        ));
        v += step;
    }
    out
}

pub fn format_number(v: f64) -> String {
    let a = v.abs();
    if a == 0.0 {
        "0".into()
    } else if !(1e-3..1e5).contains(&a) {
        format!("{v:.0e}")
    } else if a >= 100.0 {
        format!("{v:.0}")
    } else if a >= 1.0 {
        let s = format!("{v:.2}");
        s.trim_end_matches('0').trim_end_matches('.').to_owned()
    } else {
        let s = format!("{v:.3}");
        s.trim_end_matches('0').trim_end_matches('.').to_owned()
    }
}

pub fn escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ticks_are_nice_numbers() {
        let t: Vec<f64> = ticks(0.0, 1.0, false).into_iter().map(|t| t.0).collect();
        assert_eq!(t, vec![0.0, 0.2, 0.4, 0.6000000000000001, 0.8, 1.0]);
        assert_eq!(ticks(0.0, 3.0, true).len(), 4);
    }

    #[test]
    fn charts_render() {
        let svg = Chart::new("x", "y")
            .with(Series::line(
                "theory",
                vec![(0.0, 0.0), (1.0, 1.0)],
                CARDINAL,
            ))
            .with(Series::points("sim", vec![(0.5, 0.5)], GOLD))
            .svg();
        assert!(svg.starts_with("<svg") && svg.contains("polyline") && svg.contains("circle"));
    }
}
