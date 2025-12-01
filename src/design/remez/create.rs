use aether_core::real::Real;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::remez::*;
#[cfg(feature = "f128")]
use core::f128;

// Only available when we have f128 + design
#[cfg(all(feature = "f128", feature = "design"))]
use crate::remez::remez_opt::optimize_quantized_coeffs;

/* ------------ Helpers: error eval in f128 space ------------ */
#[inline]
fn max_errors_for_coeffs_f64<const N: usize, const K: usize, F>(
    rz: &RemezApprox<f128, N, K>,
    coeffs: &[f64],
    f: &F,
    a: f128,
    b: f128,
    samples: usize,
) -> (f64, f64)
where
    F: Fn(f128) -> f128,
{
    let n = samples.max(1);
    let mut max_poly = 0.0_f64;
    let mut max_func = 0.0_f64;

    for j in 0..=n {
        let t = j as f128 / n as f128;
        let x = a + (b - a) * t;

        let p_ref = rz.eval_poly(x); // f128
        let fx = f(x);               // f128

        // Horner with coeffs interpreted as f64 -> cast to f128
        let mut p_q = 0.0_f128;
        for &c in coeffs.iter().rev() {
            p_q = p_q * x + (c as f128);
        }

        let e_poly = (p_q - p_ref).abs() as f64;
        let e_func = (p_q - fx).abs() as f64;

        if e_poly > max_poly {
            max_poly = e_poly;
        }
        if e_func > max_func {
            max_func = e_func;
        }
    }

    (max_poly, max_func)
}

#[inline]
fn max_errors_for_coeffs_f32<const N: usize, const K: usize, F>(
    rz: &RemezApprox<f128, N, K>,
    coeffs: &[f32],
    f: &F,
    a: f128,
    b: f128,
    samples: usize,
) -> (f64, f64)
where
    F: Fn(f128) -> f128,
{
    let n = samples.max(1);
    let mut max_poly = 0.0_f64;
    let mut max_func = 0.0_f64;

    for j in 0..=n {
        let t = j as f128 / n as f128;
        let x = a + (b - a) * t;

        let p_ref = rz.eval_poly(x);
        let fx = f(x);

        let mut p_q = 0.0_f128;
        for &c in coeffs.iter().rev() {
            p_q = p_q * x + (c as f128);
        }

        let e_poly = (p_q - p_ref).abs() as f64;
        let e_func = (p_q - fx).abs() as f64;

        if e_poly > max_poly {
            max_poly = e_poly;
        }
        if e_func > max_func {
            max_func = e_func;
        }
    }

    (max_poly, max_func)
}

#[cfg(feature = "f16")]
#[inline]
fn max_errors_for_coeffs_f16<const N: usize, const K: usize, F>(
    rz: &RemezApprox<f128, N, K>,
    coeffs: &[f16; N],
    f: &F,
    a: f128,
    b: f128,
    samples: usize,
) -> (f64, f64)
where
    F: Fn(f128) -> f128,
{
    let n = samples.max(1);
    let mut max_poly = 0.0_f64;
    let mut max_func = 0.0_f64;

    for j in 0..=n {
        let t = j as f128 / n as f128;
        let x = a + (b - a) * t;

        let p_ref = rz.eval_poly(x);
        let fx = f(x);

        let mut p_q = 0.0_f128;
        for &c in coeffs.iter().rev() {
            let c128 = (c as f32) as f128;
            p_q = p_q * x + c128;
        }

        let e_poly = (p_q - p_ref).abs() as f64;
        let e_func = (p_q - fx).abs() as f64;

        if e_poly > max_poly {
            max_poly = e_poly;
        }
        if e_func > max_func {
            max_func = e_func;
        }
    }

    (max_poly, max_func)
}

/* ------------ RemezExport struct ------------ */
#[cfg(feature = "serde")]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct RemezExport<const N: usize, const K: usize> {
    pub function: String,
    // base f128 solution
    pub interval: (f64, f64),
    pub max_error: f64,

    // f64: cast + optimized
    pub coeffs_f64_cast: Vec<f64>,
    pub coeffs_f64_opt: Vec<f64>,
    pub max_error_f64_cast_poly: f64,
    pub max_error_f64_cast_func: f64,
    pub max_error_f64_opt_poly: f64,
    pub max_error_f64_opt_func: f64,

    // f32: cast + optimized
    pub coeffs_f32_cast: Vec<f32>,
    pub coeffs_f32_opt: Vec<f32>,
    pub max_error_f32_cast_poly: f64,
    pub max_error_f32_cast_func: f64,
    pub max_error_f32_opt_poly: f64,
    pub max_error_f32_opt_func: f64,

    // f16: cast + optimized (coeffs stored as f32)
    #[cfg(feature = "f16")]
    pub coeffs_f16_cast: Vec<f32>,
    #[cfg(feature = "f16")]
    pub coeffs_f16_opt: Vec<f32>,
    #[cfg(feature = "f16")]
    pub max_error_f16_cast_poly: f64,
    #[cfg(feature = "f16")]
    pub max_error_f16_cast_func: f64,
    #[cfg(feature = "f16")]
    pub max_error_f16_opt_poly: f64,
    #[cfg(feature = "f16")]
    pub max_error_f16_opt_func: f64,
}

/* small helper so we don't serialize NaN/inf */

#[cfg(all(feature = "serde", feature = "f128", feature = "design"))]
fn all_finite_f64(xs: &[f64]) -> bool {
    xs.iter().all(|v| v.is_finite())
}

/* ------------ Optimized path: serde + f128 + design ------------ */

#[cfg(all(feature = "serde", feature = "f128", feature = "design"))]
impl<const N: usize, const K: usize> RemezExport<N, K> {
    pub fn from_remez<F>(
        rz: &RemezApprox<f128, N, K>,
        func_name: String,
        f: F,
        interval: (f128, f128),
        max_error: f128,
    ) -> Self
    where
        F: Fn(f128) -> f128,
    {
        let (a, b) = interval;
        let a64 = a as f64;
        let b64 = b as f64;

        // sampling for error estimates
        const CAST_SAMPLES: usize = 4096;

        // SGD hyperparams
        const ITERS_64: usize = 400;
        const ITERS_32: usize = 400;
        #[cfg(feature = "f16")]
        const ITERS_16: usize = 400;

        const STEP_64: f64 = 1e-4;
        const STEP_32: f32 = 1e-4;
        #[cfg(feature = "f16")]
        const STEP_16: f32 = 1e-4;

        /* ---------- f64 path ---------- */

        // casted f64 coeffs
        let coeffs_f64_cast_arr: [f64; N] =
            core::array::from_fn(|i| rz.coefficients[i] as f64);

        let (max_error_f64_cast_poly, max_error_f64_cast_func) =
            max_errors_for_coeffs_f64(rz, &coeffs_f64_cast_arr, &f, a, b, CAST_SAMPLES);

        // optimized f64 coeffs
        let q64 = optimize_quantized_coeffs::<f64, _, N, K>(
            rz,
            &f,
            a,
            b,
            CAST_SAMPLES,
            ITERS_64,
            STEP_64,
        );

        let coeffs_f64_opt_vec: Vec<f64> = q64.coeffs.iter().copied().collect();
        let coeffs_f64_opt_are_finite = all_finite_f64(&coeffs_f64_opt_vec);

        let (coeffs_f64_opt, max_error_f64_opt_poly, max_error_f64_opt_func) =
            if coeffs_f64_opt_are_finite {
                let (e_poly_opt, e_func_opt) =
                    max_errors_for_coeffs_f64(rz, &q64.coeffs, &f, a, b, CAST_SAMPLES);
                (coeffs_f64_opt_vec, e_poly_opt, e_func_opt)
            } else {
                // fallback: mirror cast solution
                (
                    coeffs_f64_cast_arr.to_vec(),
                    max_error_f64_cast_poly,
                    max_error_f64_cast_func,
                )
            };

        let coeffs_f64_cast = coeffs_f64_cast_arr.to_vec();

        /* ---------- f32 path ---------- */

        // casted f32 coeffs
        let coeffs_f32_cast_arr: [f32; N] =
            core::array::from_fn(|i| rz.coefficients[i] as f32);

        let (max_error_f32_cast_poly, max_error_f32_cast_func) =
            max_errors_for_coeffs_f32(rz, &coeffs_f32_cast_arr, &f, a, b, CAST_SAMPLES);

        // optimized f32 coeffs
        let q32 = optimize_quantized_coeffs::<f32, _, N, K>(
            rz,
            &f,
            a,
            b,
            CAST_SAMPLES,
            ITERS_32,
            STEP_32,
        );

        let coeffs_f32_opt_vec: Vec<f32> = q32.coeffs.iter().copied().collect();
        let coeffs_f32_opt_are_finite =
            coeffs_f32_opt_vec.iter().all(|c| (*c as f64).is_finite());

        let (coeffs_f32_opt, max_error_f32_opt_poly, max_error_f32_opt_func) =
            if coeffs_f32_opt_are_finite {
                let (e_poly_opt, e_func_opt) =
                    max_errors_for_coeffs_f32(rz, &q32.coeffs, &f, a, b, CAST_SAMPLES);
                (coeffs_f32_opt_vec, e_poly_opt, e_func_opt)
            } else {
                (
                    coeffs_f32_cast_arr.to_vec(),
                    max_error_f32_cast_poly,
                    max_error_f32_cast_func,
                )
            };

        let coeffs_f32_cast = coeffs_f32_cast_arr.to_vec();

        /* ---------- f16 path (stored as f32) ---------- */

        #[cfg(feature = "f16")]
        let (
            coeffs_f16_cast,
            coeffs_f16_opt,
            max_error_f16_cast_poly,
            max_error_f16_cast_func,
            max_error_f16_opt_poly,
            max_error_f16_opt_func,
        ) = {
            use core::f16;

            // cast to f16 (stored as f32 in JSON)
            let coeffs_f16_cast_arr: [f16; N] =
                core::array::from_fn(|i| f16::from_f32(rz.coefficients[i] as f32));

            let (max_error_f16_cast_poly, max_error_f16_cast_func) =
                max_errors_for_coeffs_f16(rz, &coeffs_f16_cast_arr, &f, a, b, CAST_SAMPLES);

            // optimize in f16 space
            let q16 = optimize_quantized_coeffs::<f16, _, N, K>(
                rz,
                &f,
                a,
                b,
                CAST_SAMPLES,
                ITERS_16,
                f16::from_f32(STEP_16),
            );

            let coeffs_f16_opt_vec_f32: Vec<f32> =
                q16.coeffs.iter().map(|&c| c as f32).collect();
            let coeffs_f16_opt_are_finite =
                coeffs_f16_opt_vec_f32
                    .iter()
                    .all(|c| (*c as f64).is_finite());

            if coeffs_f16_opt_are_finite {
                let (e_poly_opt, e_func_opt) =
                    max_errors_for_coeffs_f16(rz, &q16.coeffs, &f, a, b, CAST_SAMPLES);
                (
                    coeffs_f16_cast_arr.iter().map(|c| (*c as f32)).collect(),
                    coeffs_f16_opt_vec_f32,
                    max_error_f16_cast_poly,
                    max_error_f16_cast_func,
                    e_poly_opt,
                    e_func_opt,
                )
            } else {
                (
                    coeffs_f16_cast_arr.iter().map(|c| (*c as f32)).collect(),
                    coeffs_f16_cast_arr.iter().map(|c| (*c as f32)).collect(),
                    max_error_f16_cast_poly,
                    max_error_f16_cast_func,
                    max_error_f16_cast_poly,
                    max_error_f16_cast_func,
                )
            }
        };

        Self {
            function: func_name,
            interval: (a64, b64),
            max_error: max_error as f64,

            // f64
            coeffs_f64_cast,
            coeffs_f64_opt,
            max_error_f64_cast_poly,
            max_error_f64_cast_func,
            max_error_f64_opt_poly,
            max_error_f64_opt_func,

            // f32
            coeffs_f32_cast,
            coeffs_f32_opt,
            max_error_f32_cast_poly,
            max_error_f32_cast_func,
            max_error_f32_opt_poly,
            max_error_f32_opt_func,

            // f16
            #[cfg(feature = "f16")]
            coeffs_f16_cast,
            #[cfg(feature = "f16")]
            coeffs_f16_opt,
            #[cfg(feature = "f16")]
            max_error_f16_cast_poly,
            #[cfg(feature = "f16")]
            max_error_f16_cast_func,
            #[cfg(feature = "f16")]
            max_error_f16_opt_poly,
            #[cfg(feature = "f16")]
            max_error_f16_opt_func,
        }
    }
}

/* ------------ write_coeffs_to_json ------------ */

#[cfg(all(feature = "serde", feature = "f128"))]
pub fn write_coeffs_to_json<const N: usize, const K: usize, F>(
    rz: &RemezApprox<f128, N, K>,
    func_name: &str,
    f: F,
    interval: (f128, f128),
    max_error: f128,
    file_prefix: &str,
) where
    F: Fn(f128) -> f128,
{
    use std::{fs, io::BufWriter};

    fs::create_dir_all("target/coeffs").ok();
    let path = format!("target/coeffs/{}.json", file_prefix);
    let file = fs::File::create(path).unwrap();
    let writer = BufWriter::new(file);

    let export = crate::RemezExport::<N, K>::from_remez(
        rz,
        func_name.to_string(),
        f,
        interval,
        max_error,
    );

    serde_json::to_writer_pretty(writer, &export).unwrap();
}

#[cfg(not(all(feature = "serde", feature = "f128")))]
pub fn write_coeffs_to_json<const N: usize, const K: usize, F>(
    _rz: &RemezApprox<f128, N, K>,
    _func_name: &str,
    _f: F,
    _interval: (f128, f128),
    _max_error: f128,
    _file_prefix: &str,
) where
    F: Fn(f128) -> f128,
{
    println!("Not generating coefficients file...");
}

/* ------------ plotting (unchanged logic) ------------ */

#[cfg(all(feature = "plots", feature = "f128"))]
pub fn remez_generate_plots<const N: usize, const K: usize, F>(
    rz: &RemezApprox<f128, N, K>,
    f: F,
    a: f128,
    b: f128,
    samples: usize,
    title: &str,
    prefix: &str,
) where
    F: Fn(f128) -> f128,
{
    use std::fs;
    use aether_viz::{plot_error_on_grid, plot_series, PlotStyle, XYSeries};

    fs::create_dir_all("target/plots").ok();

    let n = 3000usize;

    let a64 = a as f64;
    let b64 = b as f64;

    let xs: Vec<f64> = (0..=n)
        .map(|i| a64 + (b64 - a64) * (i as f64 / n as f64))
        .collect();

    let y_true: Vec<f64> = xs.iter().map(|&x| f(x as f128) as f64).collect();

    let y_poly: Vec<f64> = xs
        .iter()
        .map(|&x| rz.eval_poly(x as f128) as f64)
        .collect();

    let series = [
        XYSeries {
            xs: &xs,
            ys: &y_true,
            label: Some("f(x)"),
            style: PlotStyle::Line,
        },
        XYSeries {
            xs: &xs,
            ys: &y_poly,
            label: Some("Remez P(x)"),
            style: PlotStyle::Points,
        },
    ];

    let fx_file = format!("target/plots/{}_fx_vs_poly.svg", prefix);
    let err_file = format!("target/plots/{}_error.svg", prefix);

    plot_series(&series, title, Some("x"), Some("y"), Some(&fx_file)).unwrap();

    plot_error_on_grid(
        |x: f64| rz.eval_poly(x as f128) as f64,
        |x: f64| f(x as f128) as f64,
        a64,
        b64,
        samples,
        Some(&err_file),
    )
    .unwrap();
}

#[cfg(not(all(feature = "plots", feature = "f128")))]
pub fn remez_generate_plots<const N: usize, const K: usize, F>(
    _rz: &RemezApprox<f128, N, K>,
    _f: F,
    _a: f128,
    _b: f128,
    _samples: usize,
    _title: &str,
    _prefix: &str,
) where
    F: Fn(f128) -> f128,
{
    println!("Not plotting...");
}

/* ------------ Macros (unchanged interface) ------------ */

#[macro_export]
macro_rules! remez_create {
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
        $(, boundary_tol = $boundary_tol:expr, enforce_endpoints = $enforce_endpoints:expr)?
    ) => {
        pub fn $name() {
            type Rz = $crate::RemezApprox<f128, $N, $K>;

            let a64: f64 = $a;
            let b64: f64 = $b;
            let a: f128 = a64 as f128;
            let b: f128 = b64 as f128;

            let mut rz = Rz::new(a, b);
            rz.solve($f, $df);

            let max_e128 = $crate::max_error_on_grid(&rz, $f, a, b, $samples);
            let max_e64  = max_e128 as f64;
            let max_err128: f128 = $max_err as f128;

            assert!(
                max_e128 <= max_err128,
                "max |f-P| = {:.3e} over [{:.6},{:.6}]",
                max_e64, a64, b64
            );

            $(
            if $enforce_endpoints {
                let fa128 = $f(a);
                let fb128 = $f(b);
                let pa128 = rz.eval_poly(a);
                let pb128 = rz.eval_poly(b);

                let ea128 = (pa128 - fa128).abs();
                let eb128 = (pb128 - fb128).abs();

                let tol128: f128 = $boundary_tol as f128;

                let ea = ea128 as f64;
                let eb = eb128 as f64;
                let tol = tol128 as f64;

                assert!(
                    ea128 <= tol128 && eb128 <= tol128,
                    "boundary error too large: \
                     |P(a)-f(a)|={:.3e}, |P(b)-f(b)|={:.3e}, tol={:.3e}",
                    ea, eb, tol
                );
            }
            )?

            $crate::write_coeffs_to_json::<$N, $K, _>(
                &rz,
                stringify!($f),
                $f,
                (a, b),
                max_e128,
                $prefix,
            );

            $crate::remez_generate_plots::<$N, $K, _>(
                &rz,
                $f,
                a,
                b,
                $samples,
                $title,
                $prefix,
            );
        }
    };
}

#[macro_export]
macro_rules! remez_create_opt {
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
        $(, boundary_tol = $boundary_tol:expr, enforce_endpoints = $enforce_endpoints:expr)?
    ) => {
        $crate::remez_create! {
            name        = $name,
            N           = $N,
            K           = $K,
            func        = $f,
            deriv       = $df,
            interval    = [$a, $b],
            max_error   = $max_err,
            samples     = $samples,
            title       = $title,
            file_prefix = $prefix
            $(, boundary_tol = $boundary_tol,
               enforce_endpoints = $enforce_endpoints)?
        }
    };
}

#[macro_export]
macro_rules! remez_piecewise {
    (
        name  = $name:ident,
        func  = $f:path,
        deriv = $df:path,
        pieces = [
            $(
                {
                    seg_name     = $seg_name:ident,
                    N            = $N:expr,
                    K            = $K:expr,
                    interval     = [$a:expr, $b:expr],
                    max_error    = $max_err:expr,
                    samples      = $samples:expr,
                    file_prefix  = $prefix:expr
                    $(, boundary_tol = $boundary_tol:expr,
                       enforce_endpoints = $enforce_endpoints:expr)?
                }
            ),+ $(,)?
        ]
    ) => {
        $(
            $crate::remez_create_opt! {
                name        = $seg_name,
                N           = $N,
                K           = $K,
                func        = $f,
                deriv       = $df,
                interval    = [$a, $b],
                max_error   = $max_err,
                samples     = $samples,
                title       = concat!(stringify!($f), " on [", stringify!($a), ", ", stringify!($b), "]"),
                file_prefix = $prefix
                $(, boundary_tol = $boundary_tol,
                   enforce_endpoints = $enforce_endpoints)?
            }
        )+

        pub fn $name() {
            $(
                $seg_name();
            )+
        }
    };
}
