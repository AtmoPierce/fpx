#![feature(f128)]
use aether_core::real::Real;
#[cfg(feature = "design")]
use fpx::{remez_create, remez_piecewise};

fn sin_f(x: f128)->f128 { x.sin() }
fn cos_f(x: f128)->f128 { x.cos() }
fn nsin_f(x: f128)->f128 { -x.sin() }
fn tan_f(x: f128)->f128 { x.tan() }
fn secsq_f(x: f128)->f128 { 1.0 + x.tan()*x.tan()}
fn exp_f(x: f128)->f128 { x.exp()}

#[cfg(all(feature = "design", feature = "f128", feature = "plots"))]
fn design() {
    println!("Designing Coefficients");
    println!("Generating sin_f");
    remez_create!{
        name        = remez_sin,
        N           = 24,
        K           = 25,
        func        = sin_f,
        deriv       = cos_f,
        interval    = [0.0, 2.0*std::f64::consts::PI],
        max_error   = 0.00001,
        samples     = 10000,
        title       = "Remez sin(x) on [0, 2PI]",
        file_prefix = "remez_sin",
        boundary_tol = 0.00001,
        enforce_endpoints = true
    }
    remez_sin();

    remez_create!{
        name        = remez_cos,
        N           = 24,
        K           = 25,
        func        = cos_f,
        deriv       = nsin_f,
        interval    = [0.0, 2.0*std::f64::consts::PI],
        max_error   = 0.00001,
        samples     = 10000,
        title       = "Remez cos(x) on [0, 2PI]",
        file_prefix = "remez_cos",
        boundary_tol = 0.00001,
        enforce_endpoints = true
    }
    remez_cos();

    remez_create!{
        name        = remez_tan,
        N           = 30,
        K           = 31,
        func        = tan_f,
        deriv       = secsq_f,
        interval    = [-1.0*std::f64::consts::FRAC_PI_4, std::f64::consts::FRAC_PI_4],
        max_error   = 0.00001,
        samples     = 10000,
        title       = "Remez tan(x) on [-PI/2, PI/2]",
        file_prefix = "remez_tan",
        boundary_tol = 0.00001,
        enforce_endpoints = true
    }
    remez_tan();

    remez_create!{
        name        = remez_exp,
        N           = 30,
        K           = 31,
        func        = exp_f,
        deriv       = exp_f,
        interval    = [-5.0, core::f64::consts::E],
        max_error   = 0.00001,
        samples     = 10000,
        title       = "Remez exp(x) on [-PI/2, PI/2]",
        file_prefix = "remez_exp",
        boundary_tol = 0.00001,
        enforce_endpoints = true
    }
    remez_exp();
}

fn main() {
    #[cfg(all(feature = "design", feature = "f128", feature = "plots"))]
    design();
}
