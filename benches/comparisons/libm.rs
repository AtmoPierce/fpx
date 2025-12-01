#[cfg(feature = "design")]
#[cfg(test)]
mod libm {
    use crate::design::remez::{RemezApprox, max_error_on_grid};
    use core::f64::consts::{FRAC_PI_4, PI};

    #[cfg(feature = "plots")]
    use aether_viz::*;
    #[cfg(feature = "plots")]
    macro_rules! plots { ($($tt:tt)*) => { $($tt)* } }
    #[cfg(not(feature = "plots"))]
    macro_rules! plots { ($($tt:tt)*) => {}; }

    // trig
    fn sin_f(x: f64) -> f64 { x.sin() }
    fn sin_df(x: f64) -> f64 { x.cos() }

    fn cos_f(x: f64) -> f64 { x.cos() }
    fn cos_df(x: f64) -> f64 { -x.sin() }

    fn tan_f(x: f64) -> f64 { x.tan() }
    fn tan_df(x: f64) -> f64 { 1.0 / x.cos().powi(2) } // sec^2

    // inverse trig
    fn asin_f(x: f64) -> f64 { x.asin() }
    fn asin_df(x: f64) -> f64 { 1.0 / (1.0 - x * x).sqrt() }

    fn acos_f(x: f64) -> f64 { x.acos() }
    fn acos_df(x: f64) -> f64 { -1.0 / (1.0 - x * x).sqrt() }

    fn atan_f(x: f64) -> f64 { x.atan() }
    fn atan_df(x: f64) -> f64 { 1.0 / (1.0 + x * x) }

    // hyperbolic
    fn sinh_f(x: f64) -> f64 { x.sinh() }
    fn sinh_df(x: f64) -> f64 { x.cosh() }

    fn cosh_f(x: f64) -> f64 { x.cosh() }
    fn cosh_df(x: f64) -> f64 { x.sinh() }

    fn tanh_f(x: f64) -> f64 { x.tanh() }
    fn tanh_df(x: f64) -> f64 {
        // d/dx tanh = 1 / cosh^2 = 1 - tanh^2
        let t = x.tanh();
        1.0 - t * t
    }

    // exp family
    fn exp_f(x: f64) -> f64 { x.exp() }
    fn exp_df(x: f64) -> f64 { x.exp() }

    fn exp2_f(x: f64) -> f64 { x.exp2() }
    fn exp2_df(x: f64) -> f64 {
        // d/dx 2^x = ln(2) * 2^x
        core::f64::consts::LN_2 * x.exp2()
    }

    fn expm1_f(x: f64) -> f64 { x.exp_m1() }
    fn expm1_df(x: f64) -> f64 { x.exp() } // derivative is still e^x

    // ---- generic test macro ----
    macro_rules! remez_test {
        (
            name        = $name:ident,
            N           = $N:expr,
            K           = $K:expr,
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

                    let fx_file  = format!("target/plots/{}_fx_vs_poly.svg", $prefix);
                    let err_file = format!("target/plots/{}_error.svg",        $prefix);

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

    // ---- trig ----

    remez_test! {
        name        = remez_sin_minus_pi4_to_pi4,
        N           = 20,
        K           = 21,
        func        = sin_f,
        deriv       = sin_df,
        interval    = [-FRAC_PI_4, FRAC_PI_4],
        max_error   = 1e-12,
        samples     = 8192,
        title       = "Remez sin(x), n=13 on [-pi/4, pi/4]",
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
        title       = "Remez cos(x), n=12 on [-pi/4, pi/4]",
        file_prefix = "remez_cos"
    }

    remez_test! {
        name        = remez_tan_minus_pi8_to_pi8,
        N           = 20,
        K           = 21,
        func        = tan_f,
        deriv       = tan_df,
        interval    = [-PI/8.0, PI/8.0],
        max_error   = 1e-12,
        samples     = 8192,
        title       = "Remez tan(x), n=13 on [-pi/8, pi/8]",
        file_prefix = "remez_tan"
    }

    // ---- inverse trig ----

    remez_test! {
        name        = remez_asin_m1_to_1,
        N           = 25,
        K           = 26,
        func        = asin_f,
        deriv       = asin_df,
        interval    = [-1.0, 1.0],
        max_error   = 1e-12,
        samples     = 8192,
        title       = "Remez asin(x) on [-1, 1]",
        file_prefix = "remez_asin"
    }

    remez_test! {
        name        = remez_acos_m1_to_1,
        N           = 20,
        K           = 21,
        func        = acos_f,
        deriv       = acos_df,
        interval    = [-1.0, 1.0],
        max_error   = 1e-12,
        samples     = 8192,
        title       = "Remez acos(x) on [-1, 1]",
        file_prefix = "remez_acos"
    }

    remez_test! {
        name        = remez_atan_m1_to_1,
        N           = 14,
        K           = 15,
        func        = atan_f,
        deriv       = atan_df,
        interval    = [-1.0, 1.0],
        max_error   = 1e-12,
        samples     = 8192,
        title       = "Remez atan(x) on [-1, 1]",
        file_prefix = "remez_atan"
    }

    // ---- hyperbolic ----
    remez_test! {
        name        = remez_sinh_minus3_to_3,
        N           = 20,
        K           = 21,
        func        = sinh_f,
        deriv       = sinh_df,
        interval    = [-3.0, 3.0],
        max_error   = 1e-12,
        samples     = 8192,
        title       = "Remez sinh(x) on [-3, 3]",
        file_prefix = "remez_sinh"
    }

    remez_test! {
        name        = remez_cosh_0_to_3,
        N           = 20,
        K           = 21,
        func        = cosh_f,
        deriv       = cosh_df,
        interval    = [0.0, 3.0],
        max_error   = 1e-12,
        samples     = 8192,
        title       = "Remez cosh(x) on [0, 3]",
        file_prefix = "remez_cosh"
    }

    remez_test! {
        name        = remez_tanh_minus3_to_3,
        N           = 20,
        K           = 21,
        func        = tanh_f,
        deriv       = tanh_df,
        interval    = [-3.0, 3.0],
        max_error   = 1e-12,
        samples     = 8192,
        title       = "Remez tanh(x) on [-3, 3]",
        file_prefix = "remez_tanh"
    }

    // ---- exp family ----
    remez_test! {
        name        = remez_exp_minus1_to_1,
        N           = 20,
        K           = 21,
        func        = exp_f,
        deriv       = exp_df,
        interval    = [-1.0, 1.0],
        max_error   = 1e-12,
        samples     = 8192,
        title       = "Remez exp(x) on [-1, 1]",
        file_prefix = "remez_exp"
    }

    remez_test! {
        name        = remez_exp2_minus1_to_1,
        N           = 20,
        K           = 21,
        func        = exp2_f,
        deriv       = exp2_df,
        interval    = [-1.0, 1.0],
        max_error   = 1e-12,
        samples     = 8192,
        title       = "Remez exp2(x) on [-1, 1]",
        file_prefix = "remez_exp2"
    }

    remez_test! {
        name        = remez_expm1_minus1_to_1,
        N           = 20,
        K           = 21,
        func        = expm1_f,
        deriv       = expm1_df,
        interval    = [-1.0, 1.0],
        max_error   = 1e-12,
        samples     = 8192,
        title       = "Remez expm1(x) on [-1, 1]",
        file_prefix = "remez_expm1"
    }
}
