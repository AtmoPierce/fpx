use aether_core::math::{Matrix, Vector};
use aether_core::numerical_methods::solvers::lu::*;
use aether_core::numerical_methods::solvers::newton::*;
use aether_core::real::Real;
use core::fmt::Debug;

/// N = number of coefficients (degree = N-1)
/// K = number of alternation points / abscissas (must be N+1)
pub struct RemezApprox<T: Real, const N: usize, const K: usize> {
    pub coefficients: [T; N],       // monomial basis c0..c_{N-1}
    pub lower_bound: T,
    pub upper_bound: T,
    pub max_error_approximation: T, // |E|
}

impl<T: Real + Debug, const N: usize, const K: usize> RemezApprox<T, N, K> {
    pub fn new(lower: T, upper: T) -> Self {
        debug_assert!(K == N + 1, "Remez requires K = N + 1 (n+2 points)");
        Self {
            coefficients: [T::ZERO; N],
            lower_bound: lower,
            upper_bound: upper,
            max_error_approximation: T::ZERO,
        }
    }

    /// Analytic Remez (needs f and f'); single-point exchange per iteration.
    pub fn solve(&mut self, f: fn(T) -> T, df: fn(T) -> T) {
        const MAX_ITERS: usize = 100; // safeguard

        let tol   = T::from_f64(1e-7);
        let delta = T::from_f64(1e-3);
        let h0    = T::from_f64(1e-5);
        let two   = T::from_f64(2.0);

        let mut xs = self.init_cheb_extrema(); // K nodes in [a,b], ascending
        let mut newton = NewtonRaphson::new(tol, 50);

        // match the paper’s sign pattern: -E at x0, then + - + ...
        let start_neg = true;

        for _ in 0..MAX_ITERS {
            let E = self.solve_coeffs_and_E(&xs, f, start_neg);

            let mut worst_x = xs[0];
            let mut worst_e = T::ZERO;

            self.scan_alternating_gaps(&xs, start_neg, |left, right| {
                // g(x) = P'(x) - f'(x)
                let g  = |x: T| self.eval_poly_deriv(x) - df(x);
                let gp = |x: T| {
                    let hh = (T::ONE + x.abs()) * h0;
                    (g(x + hh) - g(x - hh)) / (two * hh)
                };

                let mid = (left + right) / two;
                if let Some(xstar) = newton.solve(mid, g, gp).ok() {
                    if xstar > left && xstar < right {
                        let e = self.eval_poly(xstar) - f(xstar);
                        if e.abs() > worst_e.abs() {
                            worst_e = e;
                            worst_x = xstar;
                        }
                    }
                }
            });

            // denom = max(|E|, 1)
            let e_abs = E.abs();
            let denom = if e_abs > T::ONE { e_abs } else { T::ONE };

            if self.alternates_close_to_E(&xs, f, E, delta)
                && worst_e.abs() <= (T::ONE + delta) * denom
            {
                self.max_error_approximation = e_abs;
                break;
            }

            self.exchange_in_place(&mut xs, worst_x, f);
        }
    }

    #[inline]
    pub fn eval_poly(&self, x: T) -> T {
        // Horner form
        let mut acc = T::ZERO;
        for &c in self.coefficients.iter().rev() {
            acc = acc * x + c;
        }
        acc
    }

    #[inline]
    fn eval_poly_deriv(&self, x: T) -> T {
        let mut acc = T::ZERO;
        for k in (1..N).rev() {
            let k_t = T::from_usize(k);
            let ck = self.coefficients[k] * k_t;
            acc = acc * x + ck;
        }
        acc
    }

    // ----------------- core linear solve -----------------
    /// Solve the (KxK) system:
    ///   sum_{j=0..N-1} c_j x_i^j + s_i * E = f(x_i),  s_i alternates with start_neg.
    #[allow(non_snake_case)]
    fn solve_coeffs_and_E(&mut self, xs: &[T; K], f: fn(T) -> T, start_neg: bool) -> T {
        // avoid Matrix::zeros / Vector::zeros which require Float
        let mut A = Matrix::<T, K, K>::new([[T::ZERO; K]; K]);
        let mut b = Vector::<T, K>::new([T::ZERO; K]);

        for i in 0..K {
            // monomial columns
            let mut xpow = T::ONE;
            for j in 0..N {
                A[(i, j)] = xpow;
                xpow = xpow * xs[i];
            }

            // E column
            let base = if (i & 1) == 0 { T::ONE } else { -T::ONE };
            A[(i, K - 1)] = if start_neg { -base } else { base };

            b[i] = f(xs[i]);
        }

        let sol: Vector<T, K> = A.solve(&b).expect("Remez system solve failed");

        for j in 0..N {
            self.coefficients[j] = sol[j];
        }

        sol[K - 1] // E
    }

    fn init_cheb_extrema(&self) -> [T; K] {
        let a = self.lower_bound;
        let b = self.upper_bound;

        let two = T::from_f64(2.0);

        let m = (a + b) / two;
        let r = (b - a) / two;
        let pi = T::PI;

        let mut xs = [T::ZERO; K];

        // theta from PI down to 0 -> cos in [-1..+1] -> x in [a..b], ascending
        for i in 0..K {
            let num = T::from_usize(K - 1 - i);
            let den = T::from_usize(K - 1);
            let theta = pi * num / den;
            xs[i] = m + r * theta.cos();
        }
        xs
    }

    #[allow(non_snake_case)]
    fn alternates_close_to_E(
        &self,
        xs: &[T; K],
        f: fn(T) -> T,
        E: T,
        rel_tol: T,
    ) -> bool {
        let e_abs = E.abs();
        let denom = if e_abs > T::ONE { e_abs } else { T::ONE };

        for i in 0..K {
            let e = self.eval_poly(xs[i]) - f(xs[i]);
            let want = if (i & 1) == 0 { E } else { -E };
            if (e - want).abs() > rel_tol * denom {
                return false;
            }
        }
        true
    }

    fn scan_alternating_gaps<F: FnMut(T, T)>(&self, xs: &[T; K], start_neg: bool, mut cb: F) {
        let want_even = start_neg;
        let mut done = 0usize;
        for i in 0..(K - 1) {
            if ((i & 1) == 0) != want_even {
                continue;
            }
            if done >= K - 2 {
                break;
            }
            cb(xs[i], xs[i + 1]);
            done += 1;
        }
    }

    fn exchange_in_place(&self, xs: &mut [T; K], x_new: T, f: fn(T) -> T) {
        // locate insertion index
        let mut idx = 0;
        while idx < K && xs[idx] < x_new {
            idx += 1;
        }
        let left  = if idx == 0 { 0 } else { idx - 1 };
        let right = if idx >= K { K - 1 } else { idx };

        // never drop endpoints
        let drop_candidate = if left == 0 {
            right
        } else if right == K - 1 {
            left
        } else {
            let e_left  = (self.eval_poly(xs[left])  - f(xs[left])).abs();
            let e_right = (self.eval_poly(xs[right]) - f(xs[right])).abs();
            if e_left <= e_right { left } else { right }
        };

        let mut put = drop_candidate;
        xs[put] = x_new;

        while put > 0 && xs[put] < xs[put - 1] {
            xs.swap(put, put - 1);
            put -= 1;
        }
        while put + 1 < K && xs[put] > xs[put + 1] {
            xs.swap(put, put + 1);
            put += 1;
        }
    }
}

#[derive(Clone, Copy)]
pub struct CycleReport<T: Real, const N: usize, const K: usize> {
    pub xs: [T; K],     // x_i (input nodes)
    pub coeffs: [T; N], // b_i
    pub m: [T; K],      // next extrema (endpoints pinned)
    pub E: T,           // common error magnitude/sign
}

impl<T: Real + core::fmt::Display, const N: usize, const K: usize> CycleReport<T, N, K> {
    pub fn print(&self, cycle: usize) {
        println!("Cycle {}", cycle);
        println!("E = {:+.10}", self.E);
        println!("{:<3} {:>14} {:>16} {:>16} {:>16}", "i", "x_i", "b_i", "m_i", "E_i");
        for i in 0..K {
            let bi = if i < N {
                format!("{:+.10}", self.coeffs[i])
            } else {
                String::new()
            };
            let mi = format!("{:+.10}", self.m[i]);
            let Ei = if (i & 1) == 0 { self.E } else { -self.E };
            println!(
                "{:<3} {:>14.10} {:>16} {:>16} {:>16.10}",
                i, self.xs[i], bi, mi, Ei
            );
        }
        println!();
    }
}

impl<T: Real + Debug, const N: usize, const K: usize> RemezApprox<T, N, K> {
    // recreate paper's output table
    pub fn do_cycle(
        &mut self,
        xs: &[T; K],
        f: fn(T) -> T,
        df: fn(T) -> T,
        start_neg: bool,
    ) -> CycleReport<T, N, K> {
        let E = self.solve_coeffs_and_E(xs, f, start_neg);

        let a = self.lower_bound;
        let b = self.upper_bound;

        let two = T::from_f64(2.0);
        let tol = T::from_f64(1e-7);
        let h0  = T::from_f64(1e-5);

        let mut m = [T::ZERO; K];
        m[0] = a;
        m[K - 1] = b;

        let mut newton = NewtonRaphson::new(tol, 50);

        self.scan_alternating_gaps(xs, start_neg, |left, right| {
            let g  = |x: T| self.eval_poly_deriv(x) - df(x);
            let gp = |x: T| {
                let hh = (T::ONE + x.abs()) * h0;
                (g(x + hh) - g(x - hh)) / (two * hh)
            };

            let mid = (left + right) / two;
            let xstar = newton
                .solve(mid, g, gp)
                .ok()
                .filter(|&xr| xr > left && xr < right)
                .unwrap_or(mid);

            let mut j = 1usize;
            while j < K - 1 && m[j] != T::ZERO {
                j += 1;
            }
            if j < K - 1 {
                m[j] = xstar;
            }
        });

        CycleReport {
            xs: *xs,
            coeffs: self.coefficients,
            m,
            E,
        }
    }
}

pub fn max_error_on_grid<T, const N: usize, const K: usize>(
    rz: &RemezApprox<T, N, K>,
    f: fn(T) -> T,
    a: T,
    b: T,
    samples: usize,
) -> T
where
    T: Real,
{
    let mut max_e = T::ZERO;
    for i in 0..=samples {
        let t = T::from_usize(i) / T::from_usize(samples);
        let x = a + (b - a) * t;
        let e = (rz.eval_poly(x) - f(x)).abs();
        if e > max_e {
            max_e = e;
        }
    }
    max_e
}
