//! Provides implementations of fourth-order Runge-Kutta (RK4) differential
//! equation solving with variants for adaptive step size.

use std::{
    ops::{
        Add,
        Sub,
        Mul,
    },
};
use num_complex::{
    Complex32 as C32,
    Complex64 as C64,
};
use ndarray as nd;
use crate::mkerr;

mkerr!(
    SolveError : {
        RKAErrorBound => "rka: error bound could not be satisfied",
    }
);
type SolveResult<T> = Result<T, SolveError>;

/// Provides a method to estimate the ratio between truncation errors at
/// different step sizes for a fourth-order Runge-Kutta scheme.
pub trait ErrorRatio {
    fn error_ratio(&self, other: &Self, err0: f64) -> f64;
}

impl ErrorRatio for f32 {
    fn error_ratio(&self, other: &f32, err0: f64) -> f64 {
        let scale: f64 = err0 * (self.abs() as f64 + other.abs() as f64) / 2.0;
        let diff: f64 = (*self - *other).abs() as f64;
        return diff / (scale + f64::EPSILON);
    }
}

impl ErrorRatio for f64 {
    fn error_ratio(&self, other: &f64, err0: f64) -> f64 {
        let scale: f64 = err0 * (self.abs() + other.abs()) / 2.0;
        let diff: f64 = (*self - *other).abs();
        return diff / (scale + f64::EPSILON);
    }
}

impl ErrorRatio for C32 {
    fn error_ratio(&self, other: &C32, err0: f64) -> f64 {
        let scale: f64
            = err0 * (self.norm() as f64 + other.norm() as f64) / 2.0;
        let diff: f64 = (*self - *other).norm() as f64;
        return diff / (scale + f64::EPSILON);
    }
}

impl ErrorRatio for C64 {
    fn error_ratio(&self, other: &C64, err0: f64) -> f64 {
        let scale: f64 = err0 * (self.norm() + other.norm()) / 2.0;
        let diff: f64 = (*self - *other).norm();
        return diff / (scale + f64::EPSILON);
    }
}

impl<D> ErrorRatio for nd::Array<f32, D>
where D: nd::Dimension
{
    fn error_ratio(&self, other: &nd::Array<f32, D>, err0: f64) -> f64 {
        let scale: nd::Array<f64, D>
            = (err0 / 2.0) * (
                self.mapv(|a| a.abs() as f64)
                + other.mapv(|b| b.abs() as f64)
            );
        let diff: nd::Array<f64, D> = (self - other).mapv(|d| d.abs() as f64);
        return scale.into_iter().zip(diff)
            .map(|(s, d)| d / (s + f64::EPSILON))
            .max_by(|l, r| {
                match l.partial_cmp(r) {
                    Some(ord) => ord,
                    None => std::cmp::Ordering::Less,
                }
            }).unwrap();
    }
}

impl<D> ErrorRatio for nd::Array<f64, D>
where D: nd::Dimension
{
    fn error_ratio(&self, other: &nd::Array<f64, D>, err0: f64) -> f64 {
        let scale: nd::Array<f64, D>
            = (err0 / 2.0) * (self.mapv(f64::abs) + other.mapv(f64::abs));
        let diff: nd::Array<f64, D> = (self - other).mapv(f64::abs);
        return scale.into_iter().zip(diff)
            .map(|(s, d)| d / (s + f64::EPSILON))
            .max_by(|l, r| {
                match l.partial_cmp(r) {
                    Some(ord) => ord,
                    None => std::cmp::Ordering::Less,
                }
            }).unwrap();
    }
}

impl<D> ErrorRatio for nd::Array<C32, D>
where D: nd::Dimension
{
    fn error_ratio(&self, other: &nd::Array<C32, D>, err0: f64) -> f64 {
        let scale: nd::Array<f64, D>
            = (err0 / 2.0) * (
                self.mapv(|a| a.norm() as f64)
                + other.mapv(|b| b.norm() as f64)
            );
        let diff: nd::Array<f64, D> = (self - other).mapv(|d| d.norm() as f64);
        return scale.into_iter().zip(diff)
            .map(|(s, d)| d / (s + f64::EPSILON))
            .max_by(|l, r| {
                match l.partial_cmp(r) {
                    Some(ord) => ord,
                    None => std::cmp::Ordering::Less,
                }
            }).unwrap();
    }
}

impl<D> ErrorRatio for nd::Array<C64, D>
where D: nd::Dimension
{
    fn error_ratio(&self, other: &nd::Array<C64, D>, err0: f64) -> f64 {
        let scale: nd::Array<f64, D>
            = (self.mapv(C64::norm) + other.mapv(C64::norm)) * (err0 / 2.0);
        let diff: nd::Array<f64, D> = (self - other).mapv(C64::norm);
        return scale.into_iter().zip(diff)
            .map(|(s, d)| d / (s + f64::EPSILON))
            .max_by(|l, r| {
                match l.partial_cmp(r) {
                    Some(ord) => ord,
                    None => std::cmp::Ordering::Less,
                }
            }).unwrap();
    }
}

/// Perform the operation `a + v * b` succinctly from referenced arrays.
fn array_step<A, V, B, C, ND>(
    a: &nd::Array<A, ND>,
    v: V,
    b: &nd::Array<B, ND>,
) -> nd::Array<C, ND>
where
    A: Add<B, Output = C> + Copy,
    V: Mul<B, Output = B> + Copy,
    B: Copy,
    ND: nd::Dimension,
{
    let dim = a.raw_dim();
    return std::iter::Iterator::zip(a.iter(), b.iter())
        .map(|(ak, bk)| *ak + v * *bk)
        .collect::<nd::Array1<C>>()
        .into_shape(dim).unwrap();
}

/// Take a single RK4 step, given the RHS of a 1D -> 1D governing equation.
fn rk4_step<F, U, V>(x: U, y: V, dx: U, rhs: &F) -> V
where
    F: Fn(&U, &V) -> V,
    U:
        Add<U, Output = U>
        + Mul<V, Output = V>
        + Mul<f64, Output = U>
        + Copy,
    V: Add<V, Output = V> + Mul<f64, Output = V> + Copy,
{
    let x_half: U = x + dx * 0.5_f64;
    let k1: V = rhs(&x, &y);
    let k2: V = rhs(&x_half, &(y + dx * 0.5_f64 * k1));
    let k3: V = rhs(&x_half, &(y + dx * 0.5_f64 * k2));
    let k4: V = rhs(&(x + dx), &(y + dx * k3));
    let y_new: V = y + dx * 6.0_f64.powi(-1) * (
        k1 + (k2 + k3) * 2.0_f64 + k4
    );
    return y_new;
}

/// Take a single RK4 step, given the RHS of a ND -> ND governing equation.
fn rk4_step_nd<F, U, V, ND>(
    x: U,
    y: &nd::Array<V, ND>,
    dx: U,
    rhs: &F
) -> nd::Array<V, ND>
where
    F: Fn(&U, &nd::Array<V, ND>) -> nd::Array<V, ND>,
    U:
        Add<U, Output = U>
        + Mul<V, Output = V>
        + Mul<f64, Output = U>
        + Copy,
    V: Add<V, Output = V> + Mul<f64, Output = V> + Copy,
    ND: nd::Dimension,
{
    let x_half: U = x + dx;
    let k1: nd::Array<V, ND> = rhs(&x, y);
    let k2: nd::Array<V, ND> = rhs(&x_half, &array_step(y, dx * 0.5_f64, &k1));
    let k3: nd::Array<V, ND> = rhs(&x_half, &array_step(y, dx * 0.5_f64, &k2));
    let k4: nd::Array<V, ND> = rhs(&(x + dx), &array_step(y, dx, &k3));
    let y_next: nd::Array<V, ND>
        = y.iter()
            .zip(std::iter::Iterator::zip(
                k1.into_iter().zip(k2),
                k3.into_iter().zip(k4),
            ))
            .map(|(yk, ((k1k, k2k), (k3k, k4k)))| {
                *yk + dx * 6.0_f64.powi(-1) * (
                    k1k + (k2k + k3k) * 2.0_f64 + k4k
                )
            })
            .collect::<nd::Array1<V>>()
            .into_shape(y.raw_dim()).unwrap();
    return y_next;
}

/// Driver for full RK4 integration of a 1D -> 1D governing equation.
pub fn rk4<F, U, V>(
    z0: V,
    rhs: F,
    x: &nd::Array1<U>,
) -> nd::Array1<V>
where
    F: Fn(&U, &V) -> V,
    U:
        Add<U, Output = U>
        + Sub<U, Output = U>
        + Mul<V, Output = V>
        + Mul<f64, Output = U>
        + Copy,
    V: Add<V, Output = V> + Mul<f64, Output = V> + Copy,
{
    let NX: usize = x.len();
    let dx: nd::Array1<U>
        = x.iter().skip(1)
        .zip(x.iter().take(NX - 1))
        .map(|(xn, xp)| *xn - *xp)
        .collect();
    let mut z: Vec<V> = Vec::with_capacity(NX);
    z.push(z0);
    let mut z_prev: V = z0;
    let mut z_next: V;
    for (dxk, xk) in dx.iter().zip(x.iter().take(NX - 1)) {
        z_next = rk4_step(*xk, z_prev, *dxk, &rhs);
        z.push(z_next);
        z_prev = z_next;
    }
    return nd::Array::from_vec(z);
}

/// Driver for full RK4 integration of a vector-valued governing equation.
pub fn rk4_arr<F, U, V>(
    z0: nd::Array1<V>,
    rhs: F,
    x: &nd::Array1<U>,
) -> nd::Array2<V>
where
    F: Fn(&U, &nd::Array1<V>) -> nd::Array1<V>,
    U:
        Add<U, Output = U>
        + Sub<U, Output = U>
        + Mul<V, Output = V>
        + Mul<f64, Output = U>
        + Copy,
    V: Add<V, Output = V> + Mul<f64, Output = V> + Copy,
{
    let NX: usize = x.len();
    let dx: nd::Array1<U>
        = x.iter().skip(1)
        .zip(x.iter().take(NX - 1))
        .map(|(xn, xp)| *xn - *xp)
        .collect();
    let mut z: Vec<nd::Array1<V>> = Vec::with_capacity(NX);
    z.push(z0);
    let mut z_prev: &nd::Array1<V> = z.last().unwrap();
    let mut z_next: nd::Array1<V>;
    for (dxk, xk) in dx.iter().zip(x.iter().take(NX - 1)) {
        z_next = rk4_step_nd(*xk, z_prev, *dxk, &rhs);
        z.push(z_next);
        z_prev = z.last().unwrap();
    }
    return nd::stack(
        nd::Axis(1),
        &z.iter().map(|zk| zk.view()).collect::<Vec<nd::ArrayView1<V>>>(),
    ).unwrap();
}

/// Driver for full RK4 integration of a ND -> ND governing equation.
pub fn rk4_nd<F, U, V, ND>(
    z0: nd::Array<V, ND>,
    rhs: F,
    x: &nd::Array<U, ND>,
) -> nd::Array<V, <ND as nd::Dimension>::Larger>
where
    F: Fn(&U, &nd::Array<V, ND>) -> nd::Array<V, ND>,
    U:
        Add<U, Output = U>
        + Sub<U, Output = U>
        + Mul<V, Output = V>
        + Mul<f64, Output = U>
        + Copy,
    V: Add<V, Output = V> + Mul<f64, Output = V> + Copy,
    ND: nd::Dimension,
{
    let zshape: Vec<usize> = z0.shape().to_vec();
    let NX: usize = x.len();
    let dx: nd::Array1<U>
        = x.iter().skip(1)
        .zip(x.iter().take(NX - 1))
        .map(|(xn, xp)| *xn - *xp)
        .collect();
    let mut z: Vec<nd::Array<V, ND>> = Vec::with_capacity(NX);
    z.push(z0);
    let mut z_prev: &nd::Array<V, ND> = z.last().unwrap();
    let mut z_next: nd::Array<V, ND>;
    for (dxk, xk) in dx.iter().zip(x.iter().take(NX - 1)) {
        z_next = rk4_step_nd(*xk, z_prev, *dxk, &rhs);
        z.push(z_next);
        z_prev = z.last().unwrap();
    }
    return nd::stack(
        nd::Axis(zshape.len()),
        &z.iter().map(|zk| zk.view()).collect::<Vec<nd::ArrayView<V, ND>>>(),
    ).unwrap();
}

/// Take a single RK4 step of a certain step size, given the RHS of a 1D -> 1D
/// governing equation, retuning the new x and y values as well as the
/// recommended size of the next step determined by a step-doubling strategy.
fn rka_step<F, U, V>(x: U, y: V, dx: U, rhs: &F, err: f64)
    -> SolveResult<(U, V, U)>
where
    F: Fn(&U, &V) -> V,
    U:
        Add<U, Output = U>
        + Mul<V, Output = V>
        + Mul<f64, Output = U>
        + PartialOrd
        + Copy,
    V:
        Add<V, Output = V>
        + Mul<f64, Output = V>
        + ErrorRatio
        + Copy,
{
    // define safety numbers -- particular to rk4
    const SAFE1: f64 = 0.9;
    const SAFE2: f64 = 4.0;

    let mut dx_old: U;
    let mut dx_new: U = dx;
    let (mut dx_cond1, mut dx_cond2): (U, U);
    let (mut dx_half, mut x_half, mut x_full): (U, U, U);
    let (mut y_half, mut y_half2, mut y_full): (V, V, V);
    let mut error_ratio: f64;
    for _ in 0_usize..100 {
        // take two half-sized steps
        dx_half = dx * 0.5_f64;
        x_half = x + dx_half;
        y_half = rk4_step(x, y, dx_half, &rhs);
        y_half2 = rk4_step(x_half, y_half, dx_half, &rhs);

        // take one full-sized step
        x_full = x + dx;
        y_full = rk4_step(x, y, dx, &rhs);

        // compute the estimated local truncation error
        error_ratio = y_half2.error_ratio(&y_full, err);

        // estimate new step size (with safety factors)
        dx_old = dx_new;
        if error_ratio == 0.0 {
            dx_new = dx_old * SAFE2.powi(-1);
            continue;
        }
        dx_new = dx_old * error_ratio.powf(-0.2) * SAFE1;
        dx_cond1 = dx_old * SAFE2.powi(-1);
        dx_cond2 = dx_old * SAFE2;
        dx_new = if dx_cond1 > dx_new { dx_cond1 } else { dx_new };
        dx_new = if dx_cond2 < dx_new { dx_cond2 } else { dx_new };

        if error_ratio < 1.0 {
            return Ok((x_full, y_half2, dx_new));
        }
    }
    return Err(SolveError::RKAErrorBound);
}

/// Take a single RK4 step of a certain step size, given the RHS of a ND -> ND
/// governing equation, retuning the new x and y values as well as the
/// recommended size of the next step determined by a step-doubling strategy.
fn rka_step_nd<F, U, V, ND>(
    x: U,
    y: &nd::Array<V, ND>,
    dx: U,
    rhs: &F,
    err: f64
) -> SolveResult<(U, nd::Array<V, ND>, U)>
where
    F: Fn(&U, &nd::Array<V, ND>) -> nd::Array<V, ND>,
    U:
        Add<U, Output = U>
        + Mul<V, Output = V>
        + Mul<f64, Output = U>
        + PartialOrd
        + Copy,
    V:
        Add<V, Output = V>
        + Mul<f64, Output = V>
        + Copy,
    nd::Array<V, ND>: ErrorRatio,
    ND: nd::Dimension,
{
    // define safety numbers -- particular to rk4
    const SAFE1: f64 = 0.9;
    const SAFE2: f64 = 4.0;

    let mut dx_old: U;
    let mut dx_new: U = dx;
    let (mut dx_cond1, mut dx_cond2): (U, U);
    let (mut dx_half, mut x_half, mut x_full): (U, U, U);
    let (mut y_half, mut y_half2, mut y_full):
        (nd::Array<V, ND>, nd::Array<V, ND>, nd::Array<V, ND>);
    let mut error_ratio: f64;
    for _ in 0_usize..100 {
        // take two half-sized steps
        dx_half = dx * 0.5_f64;
        x_half = x + dx_half;
        y_half = rk4_step_nd(x, y, dx_half, &rhs);
        y_half2 = rk4_step_nd(x_half, &y_half, dx_half, &rhs);

        // take one full-sized step
        x_full = x + dx;
        y_full = rk4_step_nd(x, y, dx, &rhs);

        // compute the estimated local truncation error
        error_ratio = y_half2.error_ratio(&y_full, err);

        // estimate new step size (with safety factors)
        dx_old = dx_new;
        if error_ratio == 0.0 {
            dx_new = dx_old * SAFE2.powi(-1);
            continue;
        }
        dx_new = dx_old * error_ratio.powf(-0.2) * SAFE1;
        dx_cond1 = dx_old * SAFE2.powi(-1);
        dx_cond2 = dx_old * SAFE2;
        dx_new = if dx_cond1 > dx_new { dx_cond1 } else { dx_new };
        dx_new = if dx_cond2 < dx_new { dx_cond2 } else { dx_new };

        if error_ratio < 1.0 {
            return Ok((x_full, y_half2, dx_new));
        }
    }
    return Err(SolveError::RKAErrorBound);
}

/// Driver for full RK4 integration of a 1D -> 1D governing equation using
/// adaptive step sizes controlled by a step-doubling strategy.
pub fn rka<F, U, V>(
    z0: V,
    rhs: F,
    x_bounds: (U, U),
    dx0: U,
    epsilon: f64,
) -> SolveResult<(nd::Array1<U>, nd::Array1<V>)>
where
    F: Fn(&U, &V) -> V,
    U:
        Add<U, Output = U>
        + Sub<U, Output = U>
        + Mul<V, Output = V>
        + Mul<f64, Output = U>
        + PartialOrd
        + Copy,
    V:
        Add<V, Output = V>
        + Mul<f64, Output = V>
        + ErrorRatio
        + Copy,
{
    let mut x: Vec<U> = Vec::new();
    x.push(x_bounds.0);
    let mut x_prev: U = x_bounds.0;
    let mut x_next: U;
    let mut dx: U = dx0;
    let mut z: Vec<V> = Vec::new();
    z.push(z0);
    let mut z_prev: V = z0;
    let mut z_next: V;
    let mut step: (U, V, U);
    while x_prev < x_bounds.1 {
        dx = if dx < x_bounds.1 - x_prev { dx } else { x_bounds.1 - x_prev };
        
        step = rka_step(x_prev, z_prev, dx, &rhs, epsilon)?;
        x_next = step.0;
        z_next = step.1;
        dx = step.2;

        x.push(x_next);
        z.push(z_next);

        x_prev = x_next;
        z_prev = z_next;
    }
    return Ok((nd::Array::from_vec(x), nd::Array::from_vec(z)));
}

/// Driver for full RK4 integration of a vector-valued governing equation using
/// adaptive step sizes controlled by a step-doubling strategy.
pub fn rka_arr<F, U, V>(
    z0: nd::Array1<V>,
    rhs: F,
    x_bounds: (U, U),
    dx0: U,
    epsilon: f64,
) -> SolveResult<(nd::Array1<U>, nd::Array2<V>)>
where
    F: Fn(&U, &nd::Array1<V>) -> nd::Array1<V>,
    U:
        Add<U, Output = U>
        + Sub<U, Output = U>
        + Mul<V, Output = V>
        + Mul<f64, Output = U>
        + PartialOrd
        + Copy,
    V: Add<V, Output = V> + Mul<f64, Output = V> + Copy,
    nd::Array1<V>: ErrorRatio,
{
    let mut x: Vec<U> = Vec::new();
    x.push(x_bounds.0);
    let mut x_prev: U = x_bounds.0;
    let mut x_next: U;
    let mut dx: U = dx0;
    let mut z: Vec<nd::Array1<V>> = Vec::new();
    z.push(z0);
    let mut z_prev: &nd::Array1<V> = z.last().unwrap();
    let mut z_next: nd::Array1<V>;
    let mut step: (U, nd::Array1<V>, U);
    while x_prev < x_bounds.1 {
        dx = if dx < x_bounds.1 - x_prev { dx } else { x_bounds.1 - x_prev };

        step = rka_step_nd(x_prev, z_prev, dx, &rhs, epsilon)?;
        x_next = step.0;
        z_next = step.1;
        dx = step.2;

        x.push(x_next);
        z.push(z_next);

        x_prev = x_next;
        z_prev = z.last().unwrap();
    }
    let x: nd::Array1<U> = nd::Array::from_vec(x);
    let z: nd::Array2<V>
        = nd::stack(
            nd::Axis(1),
            &z.iter().map(|zk| zk.view()).collect::<Vec<nd::ArrayView1<V>>>(),
        ).unwrap();
    return Ok((x, z));
}

/// Driver for full RK4 integration of a ND -> ND governing equation using
/// adaptive step sizes controlled by a step-doubling strategy.
#[allow(clippy::type_complexity)]
pub fn rka_nd<F, U, V, ND>(
    z0: nd::Array<V, ND>,
    rhs: F,
    x_bounds: (U, U),
    dx0: U,
    epsilon: f64,
) -> SolveResult<(nd::Array1<U>, nd::Array<V, <ND as nd::Dimension>::Larger>)>
where
    F: Fn(&U, &nd::Array<V, ND>) -> nd::Array<V, ND>,
    U:
        Add<U, Output = U>
        + Sub<U, Output = U>
        + Mul<V, Output = V>
        + Mul<f64, Output = U>
        + PartialOrd
        + Copy,
    V: Add<V, Output = V> + Mul<f64, Output = V> + Copy,
    nd::Array<V, ND>: ErrorRatio,
    ND: nd::Dimension,
{
    let zshape: Vec<usize> = z0.shape().to_vec();
    let mut x: Vec<U> = Vec::new();
    x.push(x_bounds.0);
    let mut x_prev: U = x_bounds.0;
    let mut x_next: U;
    let mut dx: U = dx0;
    let mut z: Vec<nd::Array<V, ND>> = Vec::new();
    z.push(z0);
    let mut z_prev: &nd::Array<V, ND> = z.last().unwrap();
    let mut z_next: nd::Array<V, ND>;
    let mut step: (U, nd::Array<V, ND>, U);
    while x_prev < x_bounds.1 {
        dx = if dx < x_bounds.1 - x_prev { dx } else { x_bounds.1 - x_prev };

        step = rka_step_nd(x_prev, z_prev, dx, &rhs, epsilon)?;
        x_next = step.0;
        z_next = step.1;
        dx = step.2;

        x.push(x_next);
        z.push(z_next);

        x_prev = x_next;
        z_prev = z.last().unwrap();
    }
    let x: nd::Array1<U> = nd::Array::from_vec(x);
    let z: nd::Array<V, <ND as nd::Dimension>::Larger>
        = nd::stack(
            nd::Axis(zshape.len()),
            &z.iter()
                .map(|zk| zk.view())
                .collect::<Vec<nd::ArrayView<V, ND>>>(),
        ).unwrap();
    return Ok((x, z));
}

