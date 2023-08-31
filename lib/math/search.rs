//! Provides implementations of Newton-Raphson root-finding and golden-section
//! extremum-finding.

use std::{
    ops::{
        SubAssign,
        Div,
    },
};
use num_complex::Complex64 as C64;
use ndarray::{
    self as nd,
    linalg::Dot,
};
use ndarray_linalg::InverseInto;
use crate::mkerr;

mkerr!(
    NRError : {
        SingularJacobian => "encountered singular Jacobian matrix",
    }
);
pub type NRResult<T> = Result<T, NRError>;

const INVPHI: f64 = 0.618_033_988_749_894_9; // inverse golden ratio
const INVPHI2: f64 = 0.381_966_011_250_105_1; // inverse golden ratio squared

/// Provides methods for determining when an iteration has converged.
pub trait Epsilon {
    fn lt_eps(&self, eps: f64) -> bool;
}

impl Epsilon for C64 {
    fn lt_eps(&self, eps: f64) -> bool { self.norm() < eps }
}

impl Epsilon for f64 {
    fn lt_eps(&self, eps: f64) -> bool { self < &eps }
}

impl Epsilon for (f64, f64) {
    fn lt_eps(&self, eps: f64) -> bool { (self.1 - self.0).abs() < eps }
}

impl Epsilon for nd::Array1<f64> {
    fn lt_eps(&self, eps: f64) -> bool {
        return self.iter().map(|x| x.powi(2)).sum::<f64>().sqrt() < eps
    }
}

/// Provides methods for controlling iteration over a specific region.
pub trait Region {
    type Domain;

    fn contains(&self, x: &Self::Domain) -> bool;

    fn clamp(&self, x: Self::Domain) -> Self::Domain;
}

/// Provides methods for picking a test point based on a single parameter `r`,
/// for whatever `r` could mean.
pub trait TestPoint {
    type Output;

    fn gen_point(&self, r: f64) -> Self::Output;
}

impl TestPoint for (f64, f64) {
    type Output = f64;

    fn gen_point(&self, r: f64) -> f64 {
        let h: f64 = (self.1 - self.0).abs();
        let x0: f64 = self.0.min(self.1);
        return x0 + r * h;
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Cmp {
    L = 0,
    R = 1,
}

/// Options to control Newton-Raphson root-finding.
#[derive(Clone, Copy, Debug)]
pub struct NROptions {
    /// Convergence condition (default 1e-6)
    pub epsilon: f64,

    /// Maximum number of steps to take (default 1000)
    pub maxiters: usize,
}

impl Default for NROptions {
    fn default() -> Self { Self { epsilon: 1e-6, maxiters: 1000 } }
}

/// Find a root of a 1D -> 1D function using Newton-Raphson, provided the
/// function and its first derivative.
pub fn find_root_1d<F, DF, U, V, R>(
    z0: U,
    f: F,
    df: DF,
    region: R,
    epsilon: f64,
    maxiters: usize,
) -> U
where
    F: Fn(&U) -> V,
    DF: Fn(&U) -> V,
    U: SubAssign<V> + Copy,
    V: Div<V, Output = V> + Epsilon + Copy,
    R: Region<Domain = U>,
{
    let mut z_iter: U = z0;
    let mut dz: V;
    for _ in 0..maxiters {
        dz = f(&z_iter) / df(&z_iter);
        z_iter -= dz;
        if dz.lt_eps(epsilon) {
            return z_iter;
        }
        z_iter = region.clamp(z_iter);
    }
    return z_iter;
}

/// Find a root of a ND -> ND function using Newton-Raphson, provided the
/// function and its Jacobian.
pub fn find_root_nd<F, DF, U, V, R>(
    x0: nd::Array1<U>,
    f: F,
    df: DF,
    region: R,
    epsilon: f64,
    maxiters: usize
) -> NRResult<nd::Array1<U>>
where
    F: Fn(&nd::Array1<U>) -> nd::Array1<V>,
    DF: Fn(&nd::Array1<U>) -> nd::Array2<V>,
    U: SubAssign<V> + Copy,
    nd::Array2<V>:
        InverseInto<Output = nd::Array2<V>>
        + Dot<nd::Array1<V>, Output = nd::Array1<V>>,
    V: Copy,
    nd::Array1<V>: Epsilon,
    R: Region<Domain = nd::Array1<U>>,
{
    let mut x_iter: nd::Array1<U> = x0;
    let mut Jinv: nd::Array2<V>;
    let mut dx: nd::Array1<V>;
    for _ in 0..maxiters {
        Jinv = df(&x_iter).inv_into()
            .map_err(|_| NRError::SingularJacobian)?;
        dx = Jinv.dot(&f(&x_iter));
        x_iter.iter_mut().zip(dx.iter()).for_each(|(x, dx)| *x -= *dx);
        if dx.lt_eps(epsilon) {
            return Ok(x_iter);
        }
        x_iter = region.clamp(x_iter);
    }
    return Ok(x_iter);
}

/// Find an extremum of a 1D -> 1D function using golden section search,
/// provided a comparison function.
pub fn find_extremum_1d<F, O, U, V>(
    bracket: (U, U),
    f: F,
    cmp: O,
    epsilon: f64,
    maxiters: usize,
) -> (U, V)
where
    F: Fn(&U) -> V,
    O: Fn(&V, &V) -> Cmp,
    U: Copy,
    (U, U): Epsilon + TestPoint<Output = U>,
    V: Copy,
{
    // [x1, x2] <subset> [x0, x3] is maintained throughout iteration
    let (mut x0, mut x3): (U, U) = bracket;
    let (mut x1, mut x2): (U, U)
        = (bracket.gen_point(INVPHI2), bracket.gen_point(INVPHI));
    let (mut f1, mut f2): (V, V) = (f(&x1), f(&x2));
    for _ in 0..maxiters {
        match cmp(&f1, &f2) {
            Cmp::L => {
                if (x0, x3).lt_eps(epsilon) { return (x1, f1); }
                x3 = x2;
                x2 = x1;
                f2 = f1;
                x1 = (x0, x3).gen_point(INVPHI2);
                f1 = f(&x1);
            },
            Cmp::R => {
                if (x0, x3).lt_eps(epsilon) { return (x2, f2); }
                x0 = x1;
                x1 = x2;
                f1 = f2;
                x2 = (x0, x3).gen_point(INVPHI);
                f2 = f(&x2);
            },
        }
    }
    return match cmp(&f1, &f2) {
        Cmp::L => (x1, f1),
        Cmp::R => (x2, f2),
    };
}

/// Find a minimum of a 1D -> 1D function using golden section search.
pub fn find_minimum_1d<F, U, V>(
    bracket: (U, U),
    f: F,
    epsilon: f64,
    maxiters: usize,
) -> (U, V)
where
    F: Fn(&U) -> V,
    U: Copy,
    (U, U): Epsilon + TestPoint<Output = U>,
    V: PartialEq<V> + PartialOrd<V> + Copy,
{
    return find_extremum_1d(
        bracket,
        f,
        |l: &V, r: &V| -> Cmp { if l < r { Cmp::L } else { Cmp::R } },
        epsilon,
        maxiters,
    );
}

/// Find a maximum of a 1D -> 1D function using golden section search.
pub fn find_maximum_1d<F, U, V>(
    bracket: (U, U),
    f: F,
    epsilon: f64,
    maxiters: usize,
) -> (U, V)
where
    F: Fn(&U) -> V,
    U: Copy,
    (U, U): Epsilon + TestPoint<Output = U>,
    V: PartialEq<V> + PartialOrd<V> + Copy,
{
    return find_extremum_1d(
        bracket,
        f,
        |l: &V, r: &V| -> Cmp { if l > r { Cmp::L } else { Cmp::R } },
        epsilon,
        maxiters,
    );
}

