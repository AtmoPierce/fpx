#[cfg(test)]
mod tests {
    use super::*;
    use crate::cheb::approx::*;
    
    // Keep identity test
    #[test]
    fn identity_matches() {
        let xs = [-1.0, -0.75, -0.1, 0.0, 0.1, 0.75, 1.0];
        for n in 0..12 {
            for &x in &xs {
                let t = chebyshev_t(n, x);
                let id = if x.abs() <= 1.0 {
                    (n as f64 * x.acos()).cos()
                } else if x > 1.0 {
                    (n as f64 * x.acosh()).cosh()
                } else {
                    let v = (n as f64 * (-x).acosh()).cosh();
                    if n % 2 == 0 { v } else { -v }
                };
                assert!((t - id).abs() <= 1e-12, "n={n}, x={x}, t={t}, id={id}");
            }
        }
    }
}

#[cfg(all(test, feature = "plots"))]
mod progressive_plots {
    use super::*;
    use crate::cheb::approx::*;
    use plotters::prelude::*;
    use core::f64::consts::{FRAC_PI_2, FRAC_PI_4, PI};

    /// Draw "progression" panels for a single function: one panel per N.
    fn plot_progression_for_fn(
        fname: &str,
        title: &str,
        f: impl Fn(f64) -> f64 + Copy,
        a: f64,
        b: f64,
        y_min: f64,
        y_max: f64,
        ns: &[usize],              // e.g. &[4, 8, 12, 16, 24, 32]
        samples: usize,            // e.g. 2000
    ) -> Result<(), Box<dyn std::error::Error>> {
        let cols = 3usize;
        let rows = ((ns.len() + cols - 1) / cols).max(1);

        let height = rows as u32 * 450;
        let root = BitMapBackend::new(fname, (1400, height)).into_drawing_area();
        root.fill(&WHITE)?;

        // top title band
        let (mut top, body) = root.split_vertically(50);
        top.titled(title, ("sans-serif", 26))?;

        let areas = body.split_evenly((rows, cols));

        // helper macro: evaluate a specific const N
        macro_rules! draw_panel_n {
            ($AREA:expr, $N:literal) => {{
                let area = $AREA;
                let mut chart = ChartBuilder::on(&area)
                    .margin(12)
                    .caption(format!("N = {}", $N), ("sans-serif", 20))
                    .x_label_area_size(30)
                    .y_label_area_size(45)
                    .build_cartesian_2d(a..b, y_min..y_max)?;

                chart.configure_mesh()
                    .x_desc("x")
                    .y_desc("value")
                    .label_style(("sans-serif", 12))
                    .draw()?;

                // true
                chart.draw_series(LineSeries::new(
                    (0..samples).map(|i| {
                        let x = a + (b - a) * (i as f64) / (samples as f64 - 1.0);
                        (x, f(x))
                    }),
                    RGBColor(30, 144, 255).stroke_width(2), // blue
                ))?;

                // cheb approx
                let approx = ChebApprox::<$N>::new(cheb_coeffs::<$N>(f, a, b), a, b);
                chart.draw_series(LineSeries::new(
                    (0..samples).map(|i| {
                        let x = a + (b - a) * (i as f64) / (samples as f64 - 1.0);
                        (x, approx.eval_on_ab(x))
                    }),
                    RGBColor(220, 20, 60).stroke_width(2),  // red
                ))?;

                Ok::<_, Box<dyn std::error::Error>>(())
            }};
        }

        for (i, &n) in ns.iter().enumerate() {
            let area = &areas[i];
            match n {
                2  => draw_panel_n!(area, 2)?,
                3  => draw_panel_n!(area, 3)?,
                4  => draw_panel_n!(area, 4)?,
                5  => draw_panel_n!(area, 5)?,
                6  => draw_panel_n!(area, 6)?,
                8  => draw_panel_n!(area, 8)?,
                10 => draw_panel_n!(area, 10)?,
                12 => draw_panel_n!(area, 12)?,
                16 => draw_panel_n!(area, 16)?,
                24 => draw_panel_n!(area, 24)?,
                32 => draw_panel_n!(area, 32)?,
                _  => panic!("Add a match arm for N={n}"),
            }
        }

        Ok(())
    }

    #[test]
    fn plot_progression_examples() -> Result<(), Box<dyn std::error::Error>> {
        use core::f64::consts::{PI, FRAC_PI_2, FRAC_PI_4};

        let ns = [4, 8, 12, 16, 24, 32];
        let samples = 2000;

        // --- SIN: train & plot on [0, 2PI] ---
        plot_progression_for_fn(
            "sin_progression.png",
            "sin(x): true vs Chebyshev (N progression) - domain [0, 2PI]",
            |x| x.sin(),
            0.0, 2.0 * PI,     // x-range (training == plotting)
            -1.2, 1.2,         // y-range with margin
            &ns,
            samples,
        )?;

        // --- COS: train & plot on [0, 2PI] ---
        plot_progression_for_fn(
            "cos_progression.png",
            "cos(x): true vs Chebyshev (N progression) - domain [0, 2PI]",
            |x| x.cos(),
            0.0, 2.0 * PI,
            -1.2, 1.2,
            &ns,
            samples,
        )?;

        // --- TAN: train & plot on an open interval near (-PI/4, PI/4) without touching poles ---
        let tan_left  = -0.98 * FRAC_PI_4;
        let tan_right =  0.98 * FRAC_PI_4;
        plot_progression_for_fn(
            "tan_progression.png",
            "tan(x): true vs Chebyshev (N progression) - domain ~(-0.98PI/2, 0.98PI/2)",
            |x| x.tan(),
            tan_left, tan_right,
            -6.0, 6.0,          // choose a readable y-range; adjust as you like
            &ns,
            samples,
        )?;

        Ok(())
    }


}

#[cfg(all(test, feature = "plots"))]
mod progression_error_plots {
    use super::*;
    use plotters::prelude::*;
    use core::f64::consts::{FRAC_PI_2, FRAC_PI_4, PI};
    use crate::cheb::approx::*;

    /// Make an Nx subplot figure for one function, each panel shows error for a given N.
    /// - Coefficients are trained on [train_a, train_b]
    /// - Error is plotted on [plot_a, plot_b]
    /// - If `log_scale` is true, we plot log10(|error|) with a small floor to avoid -inf
    pub fn plot_progression_error_for_fn(
        fname: &str,
        title: &str,
        f: impl Fn(f64) -> f64 + Copy,
        train_a: f64,
        train_b: f64,
        plot_a: f64,
        plot_b: f64,
        ns: &[usize],      // e.g. &[4, 8, 12, 16, 24, 32]
        samples: usize,    // e.g. 2000
        log_scale: bool,   // true => log10(|error|)
        err_floor: f64,    // e.g. 1e-300 (for log scale)
    ) -> Result<(), Box<dyn std::error::Error>> {
        let cols = 3usize;
        let rows = ((ns.len() + cols - 1) / cols).max(1);

        let height = rows as u32 * 450;
        let root = BitMapBackend::new(fname, (1400, height)).into_drawing_area();
        root.fill(&WHITE)?;

        // reserve 50 px for the figure title
        let (mut top, body) = root.split_vertically(50);
        top.titled(title, ("sans-serif", 26))?;

        let areas = body.split_evenly((rows, cols));

        // helper macro: evaluate a specific const N
        macro_rules! draw_panel_n {
            ($AREA:expr, $N:literal) => {{
                // Train coefficients on [train_a, train_b]
                let approx = ChebApprox::<$N>::new(cheb_coeffs::<$N>(f, train_a, train_b), train_a, train_b);

                // Build error samples on [plot_a, plot_b]
                let mut pts: Vec<(f64, f64)> = Vec::with_capacity(samples);
                let mut ymin = f64::MAX;
                let mut ymax = f64::MIN;
                let dx = (plot_b - plot_a) / (samples as f64);
                for i in 0..samples {
                    let x = plot_a + (i as f64 + 0.5) * dx;

                    let fx = f(x);
                    let ax = approx.eval_on_ab(x);
                    let e  = ax - fx;

                    // Dynamic (relative) floor to suppress log spikes near zero crossings.
                    // Keeps absolute floor (err_floor) but prevents -inf and crazy spikes.
                    let rel_scale = (fx.abs() + ax.abs()).max(1.0);
                    let floor     = err_floor.max(1e-16 * rel_scale);

                    let y = if log_scale {
                        (e.abs().max(floor)).log10()
                    } else {
                        e
                    };

                    ymin = ymin.min(y);
                    ymax = ymax.max(y);
                    pts.push((x, y));
                }
                // Axis padding
                let pad = if log_scale { 0.2 } else { 0.05 * (ymax - ymin).abs().max(1e-15) };
                let (ylo, yhi) = if log_scale { (ymin - pad, ymax + pad) } else { (ymin - pad, ymax + pad) };

                let area = $AREA;
                let mut chart = ChartBuilder::on(&area)
                    .margin(12)
                    .caption(format!("N = {}", $N), ("sans-serif", 20))
                    .x_label_area_size(30)
                    .y_label_area_size(55)
                    .build_cartesian_2d(plot_a..plot_b, ylo..yhi)?;

                chart
                    .configure_mesh()
                    .x_desc("x")
                    .y_desc(if log_scale { "log10(|error|)" } else { "error" })
                    .label_style(("sans-serif", 12))
                    .draw()?;

                // Shade the "extrapolation" region outside the training interval
                let shade = RGBColor(200, 200, 200).mix(0.18);
                if train_a > plot_a {
                    chart.draw_series(std::iter::once(Rectangle::new(
                        [(plot_a, ylo), (train_a, yhi)],
                        shade.filled(),
                    )))?;
                }
                if train_b < plot_b {
                    chart.draw_series(std::iter::once(Rectangle::new(
                        [(train_b, ylo), (plot_b, yhi)],
                        shade.filled(),
                    )))?;
                }

                // Vertical lines at training boundaries (for clarity)
                chart.draw_series(std::iter::once(PathElement::new(
                    vec![(train_a, ylo), (train_a, yhi)],
                    &BLACK.mix(0.4),
                )))?;
                chart.draw_series(std::iter::once(PathElement::new(
                    vec![(train_b, ylo), (train_b, yhi)],
                    &BLACK.mix(0.4),
                )))?;

                // Error curve
                chart.draw_series(LineSeries::new(pts.into_iter(), RGBColor(220, 20, 60).stroke_width(2)))?;

                Ok::<_, Box<dyn std::error::Error>>(())
            }};
        }

        for (i, &n) in ns.iter().enumerate() {
            let area = &areas[i];
            match n {
                2  => draw_panel_n!(area, 2)?,
                3  => draw_panel_n!(area, 3)?,
                4  => draw_panel_n!(area, 4)?,
                5  => draw_panel_n!(area, 5)?,
                6  => draw_panel_n!(area, 6)?,
                8  => draw_panel_n!(area, 8)?,
                10 => draw_panel_n!(area, 10)?,
                12 => draw_panel_n!(area, 12)?,
                16 => draw_panel_n!(area, 16)?,
                24 => draw_panel_n!(area, 24)?,
                32 => draw_panel_n!(area, 32)?,
                _  => panic!("Add a match arm for N={n}"),
            }
        }

        Ok(())
    }

    #[test]
    fn plot_progression_error_examples() -> Result<(), Box<dyn std::error::Error>> {
        let ns = [4, 8, 12, 16, 24, 32];
        let samples = 2000;

        let sin_train = (0.0, 2.0 * PI);
        let sin_plot  = (0.0, 2.0 * PI);
        plot_progression_error_for_fn(
            "sin_progression_error.png",
            "sin(x): log10(|error|) - train [0.0, 2.0*PI], plot ~[0.0, 2.0*PI]",
            |x| x.sin(),
            sin_train.0, sin_train.1,
            sin_plot.0,  sin_plot.1,
            &ns, samples, true, 1e-300,
        )?;

        let cos_train = (0.0, 2.0 * PI);
        let cos_plot  = (0.0, 2.0 * PI);
        plot_progression_error_for_fn(
            "cos_progression_error.png",
            "cos(x): log10(|error|) - train [0.0, 2.0*PI], plot ~[0.0, 2.0*PI]",
            |x| x.cos(),
            cos_train.0, cos_train.1,
            cos_plot.0,  cos_plot.1,
            &ns, samples, true, 1e-300,
        )?;

        let tan_train = (-FRAC_PI_4, FRAC_PI_4);
        let tan_plot  = (-FRAC_PI_4, FRAC_PI_4);
        plot_progression_error_for_fn(
            "tan_progression_error.png",
            "tan(x): log10(|error|) - train [-6, 6], plot ~[-6, 6]",
            |x| x.tan(),
            tan_train.0, tan_train.1,
            tan_plot.0,  tan_plot.1,
            &ns, samples, true, 1e-300,
        )?;


        Ok(())
    }
}

#[cfg(test)]
mod libm_parity_tests {
    use super::*;
    use crate::cheb::approx::*;
    use core::f64::consts::{PI, FRAC_PI_4};
    use libm::{sin as lsin, cos as lcos, tan as ltan};

    /// Sweep [a,b] with `samples` points, returning:
    /// (max_abs_err, max_rel_err)  with rel err = |e|/max(|truth|, rel_floor)
    fn max_abs_rel_err_on<F: Fn(f64)->f64>(
        approx_eval: &dyn Fn(f64) -> f64,
        truth: F,
        a: f64,
        b: f64,
        samples: usize,
        rel_floor: f64,
    ) -> (f64, f64) {
        let mut max_abs = 0.0f64;
        let mut max_rel = 0.0f64;
        let n = samples.max(2);
        for i in 0..n {
            let x = a + (b - a) * (i as f64) / ((n - 1) as f64);
            let t = truth(x);
            let y = approx_eval(x);
            let e = (y - t).abs();
            max_abs = max_abs.max(e);
            let denom = t.abs().max(rel_floor);
            max_rel = max_rel.max(e / denom);
        }
        (max_abs, max_rel)
    }

    /// Check an order-N Chebyshev against libm over [a,b] with hybrid tolerances.
    fn check_fn_against_libm<const N: usize, F: Fn(f64)->f64 + Copy>(
        f: F,
        name: &str,
        a: f64,
        b: f64,
        samples: usize,
        abs_tol: f64,
        rel_tol: f64,
        rel_floor: f64,
    ) {
        let coeffs = cheb_coeffs::<N>(f, a, b);
        let approx = ChebApprox::<N>::new(coeffs, a, b);

        let eval = |x: f64| approx.eval_on_ab(x);
        let (max_abs, max_rel) = max_abs_rel_err_on(&eval, f, a, b, samples, rel_floor);

        eprintln!(
            "[{} N={}, domain=[{:.6}, {:.6}]] max_abs={:.3e} (≤ {:.1e})  max_rel={:.3e} (≤ {:.1e})",
            name, N, a, b, max_abs, abs_tol, max_rel, rel_tol
        );

        assert!(
            max_abs <= abs_tol || max_rel <= rel_tol,
            "{} N={} failed: max_abs={:.3e} (tol {:.1e}), max_rel={:.3e} (tol {:.1e})",
            name, N, max_abs, abs_tol, max_rel, rel_tol
        );
    }

    // --- SIN on [0, 2PI] ---
    #[test]
    fn cheb_vs_libm_sin() {
        // (N, abs_tol, rel_tol)
        const CASES: &[(usize, f64, f64)] = &[
            (8,  1e-3,  1e-3),
            (12, 1e-6,  1e-6),
            (16, 5e-9,  5e-9),
            (24, 5e-12, 5e-12),
            (32, 1e-13, 1e-12),
        ];
        let a = 0.0;
        let b = 2.0 * PI;
        let samples = 20_000;
        let rel_floor = 1e-12; // stabilizes relative error near zeros

        for &(n, abs_tol, rel_tol) in CASES {
            match n {
                8  => check_fn_against_libm::<8 , _>(lsin, "sin", a, b, samples, abs_tol, rel_tol, rel_floor),
                12 => check_fn_against_libm::<12, _>(lsin, "sin", a, b, samples, abs_tol, rel_tol, rel_floor),
                16 => check_fn_against_libm::<16, _>(lsin, "sin", a, b, samples, abs_tol, rel_tol, rel_floor),
                24 => check_fn_against_libm::<24, _>(lsin, "sin", a, b, samples, abs_tol, rel_tol, rel_floor),
                32 => check_fn_against_libm::<32, _>(lsin, "sin", a, b, samples, abs_tol, rel_tol, rel_floor),
                _  => unreachable!(),
            }
        }
    }

    // --- COS on [0, 2PI] ---
    #[test]
    fn cheb_vs_libm_cos() {
        // (N, abs_tol, rel_tol)
        const CASES: &[(usize, f64, f64)] = &[
            (8,  2e-3,  5e-3),   // <- relaxed for N=8 over full 0..2PI
            (12, 1e-6,  1e-6),
            (16, 5e-9,  5e-9),
            (24, 5e-12, 5e-12),
            (32, 1e-13, 1e-12),
        ];
        let a = 0.0;
        let b = 2.0 * core::f64::consts::PI;
        let samples = 20_000;
        let rel_floor = 1e-10; // a touch higher helps near zeros

        for &(n, abs_tol, rel_tol) in CASES {
            match n {
                8  => check_fn_against_libm::<8 , _>(libm::cos, "cos", a, b, samples, abs_tol, rel_tol, rel_floor),
                12 => check_fn_against_libm::<12, _>(libm::cos, "cos", a, b, samples, abs_tol, rel_tol, rel_floor),
                16 => check_fn_against_libm::<16, _>(libm::cos, "cos", a, b, samples, abs_tol, rel_tol, rel_floor),
                24 => check_fn_against_libm::<24, _>(libm::cos, "cos", a, b, samples, abs_tol, rel_tol, rel_floor),
                32 => check_fn_against_libm::<32, _>(libm::cos, "cos", a, b, samples, abs_tol, rel_tol, rel_floor),
                _  => unreachable!(),
            }
        }
    }

    // --- TAN on [-PI/4, PI/4] (safe band) ---
    #[test]
    fn cheb_vs_libm_tan() {
        const CASES: &[(usize, f64, f64)] = &[
            (8,  1e-3, 1e-3),
            (12, 5e-7, 5e-7),
            (16, 1e-8, 1e-8),
            (24, 1e-10, 1e-10),
            (32, 1e-12, 1e-12),
        ];
        let a = -FRAC_PI_4;
        let b =  FRAC_PI_4;
        let samples = 20_000;
        let rel_floor = 1e-12;

        for &(n, abs_tol, rel_tol) in CASES {
            match n {
                8  => check_fn_against_libm::<8 , _>(ltan, "tan", a, b, samples, abs_tol, rel_tol, rel_floor),
                12 => check_fn_against_libm::<12, _>(ltan, "tan", a, b, samples, abs_tol, rel_tol, rel_floor),
                16 => check_fn_against_libm::<16, _>(ltan, "tan", a, b, samples, abs_tol, rel_tol, rel_floor),
                24 => check_fn_against_libm::<24, _>(ltan, "tan", a, b, samples, abs_tol, rel_tol, rel_floor),
                32 => check_fn_against_libm::<32, _>(ltan, "tan", a, b, samples, abs_tol, rel_tol, rel_floor),
                _  => unreachable!(),
            }
        }
    }
}

#[cfg(all(test, feature = "plots"))]
mod libm_comparison_plots {
    use super::*;
    use crate::cheb::approx::*;
    use plotters::prelude::*;
    use core::f64::consts::{PI, FRAC_PI_4};
    use libm::{sin as lsin, cos as lcos, tan as ltan};

    /// Compare Chebyshev approximation to libm over domain [a,b].
    /// Produces one plot with two subplots (top = function overlay, bottom = error).
    fn plot_vs_libm<const N: usize, F>(
        fname: &str,
        title: &str,
        f: F,
        a: f64,
        b: f64,
        samples: usize,
        y_min: f64,
        y_max: f64,
    ) -> Result<(), Box<dyn std::error::Error>>
    where
        F: Fn(f64) -> f64 + Copy,
    {
        let root = BitMapBackend::new(fname, (1400, 800)).into_drawing_area();
        root.fill(&WHITE)?;
        let areas = root.split_vertically(400);

        // --- top: libm vs cheb ---
        {
            let area = &areas.0;
            let mut chart = ChartBuilder::on(area)
                .margin(12)
                .caption(format!("{title} - N={N}"), ("sans-serif", 24))
                .x_label_area_size(30)
                .y_label_area_size(45)
                .build_cartesian_2d(a..b, y_min..y_max)?;
            chart.configure_mesh().x_desc("x").y_desc("value").draw()?;

            // true libm
            chart.draw_series(LineSeries::new(
                (0..samples).map(|i| {
                    let x = a + (b - a) * (i as f64) / (samples as f64 - 1.0);
                    (x, f(x))
                }),
                &RGBColor(30, 144, 255),
            ))?
            .label("libm")
            .legend(|(x, y)| Path::new(vec![(x, y), (x + 20, y)], &RGBColor(30, 144, 255)));

            // cheb
            let cheb = ChebApprox::<N>::new(cheb_coeffs::<N>(f, a, b), a, b);
            chart.draw_series(LineSeries::new(
                (0..samples).map(|i| {
                    let x = a + (b - a) * (i as f64) / (samples as f64 - 1.0);
                    (x, cheb.eval_on_ab(x))
                }),
                &RGBColor(220, 20, 60),
            ))?
            .label("Chebyshev")
            .legend(|(x, y)| Path::new(vec![(x, y), (x + 20, y)], &RGBColor(220, 20, 60)));

            chart.configure_series_labels().background_style(&WHITE.mix(0.8)).draw()?;
        }

        // --- bottom: absolute error ---
        {
            let area = &areas.1;
            let mut chart = ChartBuilder::on(area)
                .margin(12)
                .x_label_area_size(30)
                .y_label_area_size(45)
                .build_cartesian_2d(a..b, -14f64..0f64)?;
            chart.configure_mesh().x_desc("x").y_desc("log10(|error|)").draw()?;

            let cheb = ChebApprox::<N>::new(cheb_coeffs::<N>(f, a, b), a, b);
            chart.draw_series(LineSeries::new(
                (0..samples).map(|i| {
                    let x = a + (b - a) * (i as f64) / (samples as f64 - 1.0);
                    let err = (cheb.eval_on_ab(x) - f(x)).abs().max(1e-14);
                    (x, err.log10())
                }),
                &RGBColor(255, 140, 0),
            ))?;
        }

        Ok(())
    }

    #[test]
    fn plot_libm_vs_cheb() -> Result<(), Box<dyn std::error::Error>> {
        let samples = 4000;

        // sin
        plot_vs_libm::<12, _>(
            "sin_libm_vs_fpx.png",
            "sin(x): libm vs fpx",
            lsin,
            0.0,
            2.0 * PI,
            samples,
            -1.2,
            1.2,
        )?;

        // cos
        plot_vs_libm::<12, _>(
            "cos_libm_vs_fpx.png",
            "cos(x): libm vs fpx",
            lcos,
            0.0,
            2.0 * PI,
            samples,
            -1.2,
            1.2,
        )?;

        // tan (safe domain)
        plot_vs_libm::<12, _>(
            "tan_libm_vs_fpx.png",
            "tan(x): libm vs fpx",
            ltan,
            -1.0 * FRAC_PI_4,
            1.0 * FRAC_PI_4,
            samples,
            -6.0,
            6.0,
        )?;

        Ok(())
    }
}
