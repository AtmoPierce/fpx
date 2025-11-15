#[cfg(test)]
mod trig {
    use crate::remez::RemezApprox;
    use core::f64::consts::{FRAC_PI_4, PI};

    #[cfg(feature = "plots")]
    use aether_viz::*;
    #[cfg(feature = "plots")]
    macro_rules! plots { ($($tt:tt)*) => { $($tt)* } }
    #[cfg(not(feature = "plots"))]
    macro_rules! plots { ($($tt:tt)*) => {}; }

    fn max_error_on_grid<const N: usize, const K: usize>(
        rz: &RemezApprox<f64, N, K>,
        f: fn(f64) -> f64,
        a: f64,
        b: f64,
        samples: usize,
    ) -> f64 {
        let mut max_e = 0.0;
        for i in 0..=samples {
            let t = i as f64 / samples as f64;
            let x = a + (b - a) * t;
            let e = (rz.eval_poly(x) - f(x)).abs();
            if e > max_e { max_e = e; }
        }
        max_e
    }

    // ---- trig funcs + derivatives ----
    fn sin_f(x: f64) -> f64 { x.sin() }
    fn sin_df(x: f64) -> f64 { x.cos() }

    fn cos_f(x: f64) -> f64 { x.cos() }
    fn cos_df(x: f64) -> f64 { -x.sin() }

    fn tan_f(x: f64) -> f64 { x.tan() }
    fn tan_df(x: f64) -> f64 { 1.0 / x.cos().powi(2) } // sec^2

    macro_rules! remez_test {
        (
            name        = $name:ident,
            N           = $N:expr,          // number of coefficients (const generic)
            K           = $K:expr,          // number of extrema (const generic)
            func        = $f:path,
            deriv       = $df:path,
            interval    = [$a:expr, $b:expr],
            max_error   = $max_err:expr,
            samples     = $samples:expr,
            title       = $title:expr,
            file_prefix = $prefix:expr
        ) => {
            #[test]
            fn $name() {
                type Rz = RemezApprox<f64, $N, $K>;

                let a = $a;
                let b = $b;

                let mut rz = Rz::new(a, b);
                rz.solve($f, $df);

                let max_e = max_error_on_grid(&rz, $f, a, b, $samples);
                assert!(
                    max_e < $max_err,
                    "max |f-P| = {:.3e} over [{:.6},{:.6}]",
                    max_e, a, b
                );

                plots!({
                    std::fs::create_dir_all("target/plots").ok();

                    // Plot f(x) and P(x)
                    let n = 3000usize;
                    let xs: Vec<f64> = (0..=n)
                        .map(|i| a + (b - a) * (i as f64 / n as f64))
                        .collect();
                    let y_true: Vec<f64> = xs.iter().map(|&x| $f(x)).collect();
                    let y_poly: Vec<f64> = xs.iter().map(|&x| rz.eval_poly(x)).collect();

                    let series = [
                        XYSeries { xs: &xs, ys: &y_true, label: Some("f(x)"), style: PlotStyle::Line },
                        XYSeries { xs: &xs, ys: &y_poly, label: Some("Remez P(x)"), style: PlotStyle::Points },
                    ];

                    let fx_file   = format!("target/plots/{}_fx_vs_poly.svg", $prefix);
                    let err_file  = format!("target/plots/{}_error.svg",        $prefix);

                    plot_series(
                        &series,
                        $title,
                        Some("x"),
                        Some("y"),
                        Some(&fx_file),
                    ).unwrap();

                    plot_error_on_grid(
                        |x| rz.eval_poly(x),
                        $f,
                        a,
                        b,
                        $samples,
                        Some(&err_file),
                    ).unwrap();
                });
            }
        };
    }

    remez_test! {
        name        = remez_sin_minus_pi4_to_pi4,
        N           = 14,
        K           = 15,
        func        = sin_f,
        deriv       = sin_df,
        interval    = [-FRAC_PI_4, FRAC_PI_4],
        max_error   = 1e-12,
        samples     = 8192,
        title       = "Remez sin(x), n=13 on [-π/4, π/4]",
        file_prefix = "remez_sin"
    }

    remez_test! {
        name        = remez_cos_minus_pi4_to_pi4,
        N           = 13,
        K           = 14,
        func        = cos_f,
        deriv       = cos_df,
        interval    = [-FRAC_PI_4, FRAC_PI_4],
        max_error   = 1e-12,
        samples     = 8192,
        title       = "Remez cos(x), n=12 on [-π/4, π/4]",
        file_prefix = "remez_cos"
    }

    remez_test! {
        name        = remez_tan_minus_pi8_to_pi8,
        N           = 14,
        K           = 15,
        func        = tan_f,
        deriv       = tan_df,
        interval    = [-PI/8.0, PI/8.0],
        max_error   = 1e-12,
        samples     = 8192,
        title       = "Remez tan(x), n=13 on [-π/8, π/8]",
        file_prefix = "remez_tan"
    }
}
