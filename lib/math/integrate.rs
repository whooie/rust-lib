#![allow(unused_parens)]

//! Provides functions to apply simple trapezoidal and three-point Simpson's
//! rules to integrate sampled functions represented as arrays.
//!
//! Functions are intended to compute integrals of the forms
//! ```math
//! \int_a^b y(x) \,dx
//! ```
//! (the first kind) or
//! ```math
//! \int_a^x y(x') \,dx'
//!    ,~ a \leq x \leq b
//! ```
//! (the second kind).
//!
//! Where possible, `integrate*` functions will attempt to choose between the
//! two rules to minimize truncation error, with limited success in memory
//! efficiency.

use std::{
    ops::{
        Add,
        AddAssign,
        Mul,
    },
};
use num_traits::{
    Float,
    identities::Zero,
};
use ndarray::{
    self as nd,
    s,
};

/// Apply trapezoidal rule to a 1D array sampled at even intervals.
pub fn trapz<A, X>(y: &nd::Array1<A>, dx: &X) -> A
where
    A: Clone + Add<Output = A> + Mul<X, Output = A> + Zero,
    X: Float + Mul<f64, Output = X>,
{
    let n: usize = y.len();
    return (
        y[0].clone() * (dx.clone() * 0.5)
        + y.slice(s![1..n - 1]).sum() * dx.clone()
        + y[n - 1].clone() * (dx.clone() * 0.5)
    );
}

/// Apply trapezoidal rule to a 1D array sampled at uneven intervals.
pub fn trapz_nonuniform<A, X>(y: &nd::Array1<A>, x: &nd::Array1<X>) -> A
where
    A: Clone + Add<Output = A> + AddAssign<A> + Mul<X, Output = A> + Zero,
    X: Float + Mul<f64, Output = X>,
{
    let n: usize = y.len();
    let dx: Vec<X>
        = x.iter().take(n - 1).zip(x.iter().skip(1))
        .map(|(xk, xkp1)| xkp1.clone() - xk.clone())
        .collect();
    let mut acc = A::zero();
    y.iter().take(n - 1).zip(y.iter().skip(1)).zip(dx.into_iter())
        .for_each(|((ykm1, yk), dxkm1)| {
            acc += (ykm1.clone() + yk.clone()) * (dxkm1 * 0.5);
        });
    return acc;
}

/// Apply trapezoidal rule along a single axis of a N-dimensional array with
/// even sampling intervals.
pub fn trapz_axis<A, D, X>(y: &nd::Array<A, D>, dx: &X, axis: usize)
    -> nd::Array<A, <D as nd::Dimension>::Smaller>
where
    A: Clone + Add<Output = A> + Mul<X, Output = A> + Zero,
    X: Float + Mul<f64, Output = X>,
    D: nd::RemoveAxis,
{
    let n: usize = y.shape()[axis];
    return
        y.index_axis(nd::Axis(axis), 0)
            .mapv(|yk| yk * (dx.clone() * 0.5))
        + y.slice_axis(nd::Axis(axis), nd::Slice::from(1..n - 1))
            .sum_axis(nd::Axis(axis))
            .mapv(|yk| yk * dx.clone())
        + y.index_axis(nd::Axis(axis), n - 1)
            .mapv(|yk| yk * (dx.clone() * 0.5))
    ;
}

/// Apply trapezoidal rule along a single axis of a N-dimensional array with
/// uneven sampling intervals.
pub fn trapz_axis_nonuniform<A, D, X>(
    y: &nd::Array<A, D>,
    x: &nd::Array1<X>,
    axis: usize
) -> nd::Array<A, <D as nd::Dimension>::Smaller>
where
    A: Clone + Add<Output = A> + AddAssign<A> + Mul<X, Output = A> + Zero,
    X: Float + Mul<f64, Output = X>,
    D: nd::RemoveAxis,
{
    let n: usize = y.shape()[axis];
    let ndaxis = nd::Axis(axis);
    let dx: Vec<X>
        = x.iter().take(n - 1).zip(x.iter().skip(1))
        .map(|(xk, xkp1)| xkp1.clone() - xk.clone())
        .collect();
    let mut acc: nd::Array<A, <D as nd::Dimension>::Smaller>
        = nd::Array::zeros(y.raw_dim().remove_axis(ndaxis));
    y.axis_iter(ndaxis).take(n - 1).zip(y.axis_iter(ndaxis).skip(1))
        .zip(dx.into_iter())
        .for_each(|((yk, ykp1), dxk)| {
            acc += &(
                yk.mapv(|ykj| ykj * (dxk.clone() * 0.5))
                + ykp1.mapv(|ykp1j| ykp1j * (dxk.clone() * 0.5))
            )
        });
    return acc;
}

/// Apply Simpson's rule to a 1D array sampled at even intervals.
pub fn simpson<A, X>(y: &nd::Array1<A>, dx: &X) -> A
where
    A: Clone + Add<Output = A> + Mul<X, Output = A> + Zero,
    X: Float + Mul<f64, Output = X>,
{
    let n: usize = y.len();
    return (
        y[0].clone() * (dx.clone() * (1.0 / 3.0))
        + y.slice(s![1..n - 1;2]).sum() * (dx.clone() * (4.0 / 3.0))
        + y.slice(s![2..n - 1;2]).sum() * (dx.clone() * (2.0 / 3.0))
        + y[n - 1].clone() * (dx.clone() * (1.0 / 3.0))
    );
}

/// Apply Simpson's rule along a single axis of a N-dimensional array ith even
/// sampling intervals.
pub fn simpson_axis<A, D, X>(y: &nd::Array<A, D>, dx: &X, axis: usize)
    -> nd::Array<A, <D as nd::Dimension>::Smaller>
where
    A: Clone + Add<Output = A> + Mul<X, Output = A> + Zero,
    X: Float + Mul<f64, Output = X>,
    D: nd::RemoveAxis,
{
    let n: usize = y.shape()[axis];
    return
        y.index_axis(nd::Axis(axis), 0)
            .mapv(|yk| yk * (dx.clone() * (1.0 / 3.0)))
        + y.slice_axis(nd::Axis(axis), nd::Slice::from(1..n - 1).step_by(2))
            .sum_axis(nd::Axis(axis))
            .mapv(|yk| yk * (dx.clone() * (4.0 / 3.0)))
        + y.slice_axis(nd::Axis(axis), nd::Slice::from(2..n - 1).step_by(2))
            .sum_axis(nd::Axis(axis))
            .mapv(|yk| yk * (dx.clone() * (2.0 / 3.0)))
        + y.index_axis(nd::Axis(axis), n - 1)
            .mapv(|yk| yk * (dx.clone() * (1.0 / 3.0)))
    ;
}

/// Apply either trapezoidal or Simpson's rule, depending on the number of
/// points $`n`$ sampled in the array.
///
/// For even $`n`$ use trapezoidal; for odd $`n`$ use Simpson's.
pub fn integrate<A, X>(y: &nd::Array1<A>, dx: &X) -> A
where
    A: Clone + Add<Output = A> + Mul<X, Output = A> + Zero,
    X: Float + Mul<f64, Output = X>,
{
    return if y.len() % 2 == 1 {
        simpson(y, dx)
    } else {
        trapz(y, dx)
    };
}

/// Apply either trapezoidal or Simpson's rule to a single axis, depending on
/// the size $`n`$ of the axis.
///
/// For even $`n`$ use trapezoidal; for odd $`n`$ use Simpson's.
pub fn integrate_axis<A, D, X>(y: &nd::Array<A, D>, dx: &X, axis: usize)
    -> nd::Array<A, <D as nd::Dimension>::Smaller>
where
    A: Clone + Add<Output = A> + Mul<X, Output = A> + Zero,
    X: Float + Mul<f64, Output = X>,
    D: nd::RemoveAxis,
{
    return if y.len() % 2 == 1 {
        simpson_axis(y, dx, axis)
    } else {
        trapz_axis(y, dx, axis)
    };
}

/// Compute the progressive integral via trapezoidal rule.
pub fn trapz_prog<A, X>(y: &nd::Array1<A>, dx: &X) -> nd::Array1<A>
where
    A: Clone + Add<Output = A> + AddAssign<A> + Mul<X, Output = A> + Zero,
    X: Float + Mul<f64, Output = X>,
{
    let n: usize = y.len();
    let mut I: nd::Array1<A> = nd::Array::zeros(n);
    let mut acc = A::zero();
    let i: nd::Array1<A>
        = y.iter().take(n - 1).zip(y.iter().skip(1))
        .map(|(ykm1, yk)| {
            acc += (ykm1.clone() + yk.clone()) * (dx.clone() * 0.5);
            acc.clone()
        })
        .collect();
    i.move_into(I.slice_mut(s![1..n]));
    return I;
}

/// Compute the progressive integral for uneven sampling via trapezoidal rule.
pub fn trapz_prog_nonuniform<A, X>(y: &nd::Array1<A>, x: &nd::Array1<X>)
    -> nd::Array1<A>
where
    A: Clone + Add<Output = A> + AddAssign<A> + Mul<X, Output = A> + Zero,
    X: Float + Mul<f64, Output = X>,
{
    let n: usize = y.len();
    let dx: Vec<X>
        = x.iter().take(n - 1).zip(x.iter().skip(1))
        .map(|(xk, xkp1)| xkp1.clone() - xk.clone())
        .collect();
    let mut I: nd::Array1<A> = nd::Array::zeros(n);
    let mut acc = A::zero();
    let i: nd::Array1<A>
        = y.iter().take(n - 1).zip(y.iter().skip(1)).zip(dx.into_iter())
        .map(|((ykm1, yk), dxkm1)| {
            acc += (ykm1.clone() + yk.clone()) * (dxkm1 * 0.5);
            acc.clone()
        })
        .collect();
    i.move_into(I.slice_mut(s![1..n]));
    return I;
}

/// Compute the progressive integral along a single axis via trapezoidal rule.
///
/// **Warning: this routine is *very* expensive to run!**
pub fn trapz_prog_axis<A, D, X>(y: &nd::Array<A, D>, dx: &X, axis: usize)
    -> nd::Array<A, D>
where
    A: Clone + Add<Output = A> + AddAssign<A> + Mul<X, Output = A> + Zero,
    X: Float + Mul<f64, Output = X>,
    D: nd::RemoveAxis,
    <D as nd::Dimension>::Smaller: nd::Dimension<Larger = D>,
{
    let n: usize = y.shape()[axis];
    let ndaxis = nd::Axis(axis);
    let mut I: nd::Array<A, D> = nd::Array::zeros(y.raw_dim());
    let mut acc: nd::Array<A, <D as nd::Dimension>::Smaller>
        = nd::Array::zeros(y.raw_dim().remove_axis(ndaxis));
    let slices: Vec<nd::Array<A, <D as nd::Dimension>::Smaller>>
        = y.axis_iter(ndaxis).take(n - 1).zip(y.axis_iter(ndaxis).skip(1))
        .map(|(ykm1, yk)| {
            acc.iter_mut().zip(ykm1.iter().zip(yk.iter()))
                .for_each(|(accj, (ykm1j, ykj))| {
                    *accj += (ykm1j.clone() + ykj.clone()) * (dx.clone() * 0.5);
                });
            acc.clone()
        })
        .collect();
    let i: nd::Array<A, D>
        = nd::stack(
            ndaxis,
            &slices.iter()
                .map(|ik| ik.view())
                .collect::<
                    Vec<nd::ArrayView<A, <D as nd::Dimension>::Smaller>>
                >()
        ).unwrap();
    i.move_into(I.slice_axis_mut(ndaxis, nd::Slice::from(1..n)));
    return I;
}

/// Compute the progressive integral along a single axis for uneven sampling
/// via trapezoidal rule.
///
/// **Warning: this routine is *very* expensive to run!**
pub fn trapz_prog_axis_nonuniform<A, D, X>(
    y: &nd::Array<A, D>,
    x: &nd::Array1<X>,
    axis: usize
) -> nd::Array<A, D>
where
    A: Clone + Add<Output = A> + AddAssign<A> + Mul<X, Output = A> + Zero,
    X: Float + Mul<f64, Output = X>,
    D: nd::RemoveAxis,
    <D as nd::Dimension>::Smaller: nd::Dimension<Larger = D>,
{
    let n: usize = y.shape()[axis];
    let ndaxis = nd::Axis(axis);
    let dx: Vec<X>
        = x.iter().take(n - 1).zip(x.iter().skip(1))
        .map(|(xk, xkp1)| xkp1.clone() - xk.clone())
        .collect();
    let mut I: nd::Array<A, D> = nd::Array::zeros(y.raw_dim());
    let mut acc: nd::Array<A, <D as nd::Dimension>::Smaller>
        = nd::Array::zeros(y.raw_dim().remove_axis(ndaxis));
    let slices: Vec<nd::Array<A, <D as nd::Dimension>::Smaller>>
        = y.axis_iter(ndaxis).take(n - 1).zip(y.axis_iter(ndaxis).skip(1))
        .zip(dx.into_iter())
        .map(|((ykm1, yk), dxkm1)| {
            acc.iter_mut().zip(ykm1.iter().zip(yk.iter()))
                .for_each(|(accj, (ykm1j, ykj))| {
                    *accj += (
                        ykm1j.clone() + ykj.clone()
                    ) * (dxkm1.clone() * 0.5);
                });
            acc.clone()
        })
        .collect();
    let i: nd::Array<A, D>
        = nd::stack(
            ndaxis,
            &slices.iter()
                .map(|ik| ik.view())
                .collect::<
                    Vec<nd::ArrayView<A, <D as nd::Dimension>::Smaller>>
                >()
        ).unwrap();
    i.move_into(I.slice_axis_mut(ndaxis, nd::Slice::from(1..n)));
    return I;
}

fn simps_factor(km1: usize) -> f64 {
    return if km1 % 2 == 0 { 1.0 / 3.0 } else { 4.0 / 3.0 }
}

/// Compute the progressive integral via Simpson's rule.
pub fn simpson_prog<A, X>(y: &nd::Array1<A>, dx: &X) -> nd::Array1<A>
where
    A: Clone + Add<Output = A> + AddAssign<A> + Mul<X, Output = A> + Zero,
    X: Float + Mul<f64, Output = X>,
{
    let n: usize = y.len();
    let mut I: nd::Array1<A> = nd::Array::zeros(n);
    let mut acc = A::zero();
    let i: nd::Array1<A>
        = y.iter().take(n - 1).zip(y.iter().skip(1)).enumerate()
        .map(|(km1, (ykm1, yk))| {
            acc += (
                ykm1.clone() * (dx.clone() * simps_factor(km1))
                + yk.clone() * (dx.clone() * (1.0 / 3.0))
            );
            acc.clone()
        })
        .collect();
    i.move_into(I.slice_mut(s![1..n]));
    return I;
}

/// Compute the progressive integral along a single axis via Simpson's rule.
///
/// **Warning: this routine is *very* expensive to run!**
pub fn simpson_prog_axis<A, D, X>(y: &nd::Array<A, D>, dx: &X, axis: usize)
    -> nd::Array<A, D>
where
    A: Clone + Add<Output = A> + AddAssign<A> + Mul<X, Output = A> + Zero,
    X: Float + Mul<f64, Output = X>,
    D: nd::RemoveAxis,
    <D as nd::Dimension>::Smaller: nd::Dimension<Larger = D>,
{
    let n: usize = y.shape()[axis];
    let ndaxis = nd::Axis(axis);
    let mut I: nd::Array<A, D> = nd::Array::zeros(y.raw_dim());
    let mut acc: nd::Array<A, <D as nd::Dimension>::Smaller>
        = nd::Array::zeros(y.raw_dim().remove_axis(ndaxis));
    let slices: Vec<nd::Array<A, <D as nd::Dimension>::Smaller>>
        = y.axis_iter(ndaxis).take(n - 1).zip(y.axis_iter(ndaxis).skip(1))
        .enumerate()
        .map(|(km1, (ykm1, yk))| {
            acc.iter_mut().zip(ykm1.iter().zip(yk.iter()))
                .for_each(|(accj, (ykm1j, ykj))| {
                    *accj += (
                        ykm1j.clone() * (dx.clone() * simps_factor(km1))
                        + ykj.clone() * (dx.clone() * (1.0 / 3.0))
                    );
                });
            acc.clone()
        })
        .collect();
    let i: nd::Array<A, D>
        = nd::stack(
            ndaxis,
            &slices.iter()
                .map(|ik| ik.view())
                .collect::<
                    Vec<nd::ArrayView<A, <D as nd::Dimension>::Smaller>>
                >()
        ).unwrap();
    i.move_into(I.slice_axis_mut(ndaxis, nd::Slice::from(1..n)));
    return I;
}

/// Compute the progressive integral, minimizing truncation error by calculating
/// using both rules and returning a final array that alternates between the
/// two.
pub fn integrate_prog<A, X>(y: &nd::Array1<A>, dx: &X) -> nd::Array1<A>
where
    A: Clone + Add<Output = A> + AddAssign<A> + Mul<X, Output = A> + Zero,
    X: Float + Mul<f64, Output = X>,
{
    let tr: nd::Array1<A> = trapz_prog(y, dx);
    let si: nd::Array1<A> = simpson_prog(y, dx);
    let I: nd::Array1<A>
        = tr.into_iter().zip(si.into_iter()).enumerate()
        .map(|(k, (trk, sik))| if k % 2 == 0 { sik } else { trk })
        .collect();
    return I;
}

/// Compute the progressive integral along a single axis, minimizing truncation
/// error by calculating using and retuning a final array that alternates
/// between the two.
///
/// **Warning: this routine is *extremely* expensive to run!**
pub fn integrate_prog_axis<A, D, X>(y: &nd::Array<A, D>, dx: &X, axis: usize)
    -> nd::Array<A, D>
where
    A: Clone + Add<Output = A> + AddAssign<A> + Mul<X, Output = A> + Zero,
    X: Float + Mul<f64, Output = X>,
    D: nd::RemoveAxis,
    <D as nd::Dimension>::Smaller: nd::Dimension<Larger = D>,
{
    let mut tr: nd::Array<A, D> = trapz_prog_axis(y, dx, axis);
    tr.swap_axes(0, axis);
    let mut si: nd::Array<A, D> = simpson_prog_axis(y, dx, axis);
    si.swap_axes(0, axis);
    let ndaxis = nd::Axis(0);
    let mut I: nd::Array<A, D>
        = tr.axis_iter(ndaxis).zip(si.axis_iter(ndaxis)).enumerate()
        .flat_map(|(k, (trk, sik))| {
            if k % 2 == 0 {
                sik.into_owned().into_iter()
            } else {
                trk.into_owned().into_iter()
            }
        })
        .collect::<nd::Array1<A>>()
        .into_shape(y.raw_dim()).unwrap();
    I.swap_axes(0, axis);
    return I;
}

