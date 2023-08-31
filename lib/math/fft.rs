//! Provides various functions surrounding application of the (inverse) Fast
//! Fourier Transform, wrapping around `rustfft` and `ndrustfft` functions as
//! needed.
//!
//! Also provides functions for generating frequency- and real-space coordinate
//! arrays associated with the transform.
//!
//! All transforms are complex-valued.

use rustfft as fft;
use ndrustfft as ndfft;
use num_complex::Complex64 as C64;
use ndarray::{
    self as nd,
    concatenate,
    s,
};

/// Generate a frequency-space coordinate array based on a sampling time.
pub fn fft_freq(N: usize, dt: f64) -> nd::Array1<f64> {
    return if N % 2 == 0 {
        let fp: nd::Array1<f64>
            = (0..N / 2)
            .map(|k| k as f64 / (N as f64 * dt))
            .collect();
        let fm: nd::Array1<f64>
            = (1..N / 2 + 1).rev()
            .map(|k| -(k as f64) / (N as f64 * dt))
            .collect();
        concatenate!(nd::Axis(0), fp, fm)
    } else {
        let fp: nd::Array1<f64>
            = (0..(N + 1) / 2)
            .map(|k| k as f64 / (N as f64 * dt))
            .collect();
        let fm: nd::Array1<f64>
            = (1..(N + 1) / 2).rev()
            .map(|k| -(k as f64) / (N as f64 * dt))
            .collect();
        concatenate!(nd::Axis(0), fp, fm)
    };
}

/// Perform a FFT on a 1D array.
pub fn fft(x: &nd::Array1<C64>) -> nd::Array1<C64> {
    let N: usize = x.len();
    let mut buf: Vec<C64> = x.to_vec();
    let mut planner = fft::FftPlanner::new();
    let fft_plan = planner.plan_fft_forward(N);
    fft_plan.process(&mut buf);
    return nd::Array::from(buf);
}

/// Perform a FFT on a 1D array, also generating frequency-space coordinates.
pub fn do_fft(x: &nd::Array1<C64>, dt: f64)
    -> (nd::Array1<C64>, nd::Array1<f64>)
{
    let N: usize = x.len();
    return (fft(x), fft_freq(N, dt));
}

/// Generate a copy of an array such that it begins with its negative-frequency
/// components.
pub fn fft_shifted<T>(x: &nd::Array1<T>) -> nd::Array1<T>
where T: Clone
{
    let N: usize = x.len();
    let (p, m) = if N % 2 == 0 {
        (x.slice(s![0..N / 2]), x.slice(s![N / 2..N]))
    } else {
        (x.slice(s![0..N / 2 + 1]), x.slice(s![N / 2 + 1..N]))
    };
    return concatenate!(nd::Axis(0), m.into_owned(), p.into_owned());
}

/// Rearrange the elements of an array *in place* such that it begins with its
/// negative-frequency components.
pub fn fft_shift<T>(x: nd::ArrayViewMut1<T>)
where T: Clone
{
    let N: usize = x.len();
    let (p, m) = if N % 2 == 0 {
        (x.slice(s![0..N / 2]), x.slice(s![N / 2..N]))
    } else {
        (x.slice(s![0..N / 2 + 1]), x.slice(s![N / 2 + 1..N]))
    };
    let new = concatenate!(nd::Axis(0), m.into_owned(), p.into_owned());
    new.move_into(x);
}

/// Perform a FFT along a single axis of a N-dimensional array.
pub fn fft_nd<D>(X: &nd::Array<C64, D>, axis: usize) -> nd::Array<C64, D>
where D: nd::Dimension
{
    let N: usize = X.shape()[axis];
    let mut buf: nd::Array<C64, D> = nd::Array::zeros(X.raw_dim());
    let mut handler: ndfft::FftHandler<f64> = ndfft::FftHandler::new(N);
    ndfft::ndfft(X, &mut buf, &mut handler, axis);
    return buf;
}

/// Perform a FFT along a single axis of a N-dimensional array, also generating
/// frequency-space coordinates.
pub fn do_fft_nd<D>(X: &nd::Array<C64, D>, dt: f64, axis: usize)
    -> (nd::Array<C64, D>, nd::Array1<f64>)
where D: nd::Dimension
{
    let N: usize = X.shape()[axis];
    return (fft_nd(X, axis), fft_freq(N, dt));
}

/// Generate a copy of a N-dimensional array such that it begins with its
/// negative-frequeny components along an axis.
pub fn fft_shifted_axis<T, D>(X: &nd::Array<T, D>, axis: usize)
    -> nd::Array<T, D>
where
    T: Clone,
    D: nd::Dimension + nd::RemoveAxis,
{
    let N: usize = X.shape()[axis];
    let (P, M) = if N % 2 == 0 {
        (
            X.slice_axis(nd::Axis(axis), nd::Slice::from(0..N / 2)),
            X.slice_axis(nd::Axis(axis), nd::Slice::from(N / 2..N)),
        )
    } else {
        (
            X.slice_axis(nd::Axis(axis), nd::Slice::from(0..N / 2 + 1)),
            X.slice_axis(nd::Axis(axis), nd::Slice::from(N / 2 + 1..N)),
        )
    };
    return concatenate!(nd::Axis(axis), M.into_owned(), P.into_owned());
}

/// Rearrange the elements of a N-dimensional array *in place* such it begins
/// with its negative-frequency components along an axis.
pub fn fft_shift_axis<T, D>(X: nd::ArrayViewMut<T, D>, axis: usize)
where
    T: Clone,
    D: nd::Dimension + nd::RemoveAxis,
{
    let N: usize = X.shape()[axis];
    let (P, M) = if N % 2 == 0 {
        (
            X.slice_axis(nd::Axis(axis), nd::Slice::from(0..N / 2)),
            X.slice_axis(nd::Axis(axis), nd::Slice::from(N / 2..N)),
        )
    } else {
        (
            X.slice_axis(nd::Axis(axis), nd::Slice::from(0..N / 2 + 1)),
            X.slice_axis(nd::Axis(axis), nd::Slice::from(N / 2 + 1..N)),
        )
    };
    let new = concatenate!(nd::Axis(axis), M.into_owned(), P.into_owned());
    new.move_into(X);
}

/// Generate a real-space coordinate array based on a frequency-space bin width.
pub fn ifft_coord(N: usize, df: f64) -> nd::Array1<f64> {
    return nd::Array1::range(0.0, N as f64, 1.0) / (N as f64 * df);
}

/// Perform an inverse FFT on a 1D array.
pub fn ifft(x: &nd::Array1<C64>) -> nd::Array1<C64> {
    let N: usize = x.len();
    let mut buf: Vec<C64> = x.to_vec();
    let mut planner = fft::FftPlanner::new();
    let ifft_plan = planner.plan_fft_inverse(N);
    ifft_plan.process(&mut buf);
    return nd::Array::from(buf);
}

/// Perform an inverse FFT on a 1D array, also generating real-space
/// coordinates.
pub fn do_ifft(x: &nd::Array1<C64>, df: f64)
    -> (nd::Array1<C64>, nd::Array1<f64>)
{
    let N: usize = x.len();
    return (ifft(x), ifft_coord(N, df));
}

/// Generate a copy of an array such that it begins with its zero-frequency
/// component.
pub fn fft_ishifted<T>(x: &nd::Array1<T>) -> nd::Array1<T>
where T: Clone
{
    let N: usize = x.len();
    let (m, p) = (x.slice(s![0..N / 2]), x.slice(s![N / 2..N]));
    return concatenate!(nd::Axis(0), p.into_owned(), m.into_owned());
}

/// Rearrange an array *in place* such that it begins with its zero-frequency
/// component.
pub fn fft_ishift<T>(x: nd::ArrayViewMut1<T>)
where T: Clone
{
    let N: usize = x.len();
    let (m, p) = (x.slice(s![0..N / 2]), x.slice(s![N / 2..N]));
    let new = concatenate!(nd::Axis(0), p.into_owned(), m.into_owned());
    new.move_into(x);
}

/// Perform an inverse FFT along a single axis of a N-dimensional array.
pub fn ifft_nd<D>(X: &nd::Array<C64, D>, axis: usize) -> nd::Array<C64, D>
where D: nd::Dimension
{
    let N: usize = X.shape()[axis];
    let mut buf: nd::Array<C64, D> = nd::Array::zeros(X.raw_dim());
    let mut handler: ndfft::FftHandler<f64> = ndfft::FftHandler::new(N);
    ndfft::ndifft(X, &mut buf, &mut handler, axis);
    return buf;
}

/// Perform an inverse FFT along a single axis of a N-dimensional array, also
/// generating real-space coordinates.
pub fn do_ifft_nd<D>(X: &nd::Array<C64, D>, df: f64, axis: usize)
    -> (nd::Array<C64, D>, nd::Array1<f64>)
where D: nd::Dimension
{
    let N: usize = X.shape()[axis];
    return (ifft_nd(X, axis), ifft_coord(N, df));
}

/// Generate a copy of a N-dimensional array such that it begins with its
/// zero-frequency component along an axis.
pub fn fft_ishifted_axis<T, D>(X: &nd::Array<T, D>, axis: usize)
    -> nd::Array<T, D>
where
    T: Clone,
    D: nd::Dimension + nd::RemoveAxis,
{
    let N: usize = X.shape()[axis];
    let (M, P)
        = (
            X.slice_axis(nd::Axis(axis), nd::Slice::from(0..N / 2)),
            X.slice_axis(nd::Axis(axis), nd::Slice::from(N / 2..N)),
        );
    return concatenate!(nd::Axis(axis), P.into_owned(), M.into_owned());
}

/// Rearrange the elements of a N-dimensional array *in place* such that it
/// begins with its zero-frequency components along an axis.
pub fn fft_ishift_axis<T, D>(X: nd::ArrayViewMut<T, D>, axis: usize)
where
    T: Clone,
    D: nd::Dimension + nd::RemoveAxis,
{
    let N: usize = X.shape()[axis];
    let (M, P)
        = (
            X.slice_axis(nd::Axis(axis), nd::Slice::from(0..N / 2)),
            X.slice_axis(nd::Axis(axis), nd::Slice::from(N / 2..N)),
        );
    let new = concatenate!(nd::Axis(axis), P.into_owned(), M.into_owned());
    new.move_into(X);
}

