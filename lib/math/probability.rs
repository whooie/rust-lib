#![allow(unused_parens)]

//! Provides basic tools to work with probabilities and random variables. In
//! most realistic use-cases, one should reach for [statrs] instead.
//!
//! Most of the material in this module focuses on sampling random values
//! (discrete or continuous) from arbitrary distributions provided by the user.
//! In that case, the data provided by the user is taken more-or-less at face
//! value and the mathematical properties of the data (e.g. normalization, CDF
//! values lying between zero and one) are only verified for the remaining
//! degrees of freedom.
//!
//! In the cases where egregious deviations from the formalism are encountered
//! (e.g. negative probabilities, non-monotonic CDFs), a `panic!` is preferred
//! to an `Err`, since it is generally assumed that the user knows the core
//! concepts at play.
//!
//! [statrs]: https://github.com/statrs-dev/statrs

use std::{
    collections::HashMap,
    convert::{
        TryFrom,
        TryInto,
    },
    fmt,
    ops::{
        SubAssign,
        Mul,
        Range,
        RangeFrom,
        RangeFull,
        RangeInclusive,
        RangeTo,
        RangeToInclusive,
    }
};
use num_traits::{
    Float,
    One,
};
use rand::{
    prelude as rnd,
    Rng,
};
use ndarray as nd;
use crate::{
    mkerr,
    math::{
        integrate,
        search::{
            self,
            Epsilon,
            Region,
            NROptions,
        },
    },
};

mkerr!(
    ProbError : {
        BadProbability => "probabilities must be between 0 and 1",
    }
);
pub type ProbResult<T> = Result<T, ProbError>;

/// Represents a `f64` in the range $`[0, 1]`$.
///
/// The only ways to instantiate a `Probability` require that this condition be
/// satisfied; thus it is guaranteed to hold for any such object.
#[derive(Clone, Copy, PartialEq, PartialOrd, Debug)]
pub struct Probability {
    p: f64
}

impl Eq for Probability { }

impl Ord for Probability {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // this is safe due to construction constraints
        return self.p().partial_cmp(&other.p()).unwrap();
    }
}

macro_rules! impl_probability_try_from(
    ( $type:ident ) => {
        impl TryFrom<$type> for Probability {
            type Error = ProbError;

            fn try_from(p: $type) -> ProbResult<Self> {
                return if (0.0..=1.0).contains(&p) {
                    Ok(Probability { p: p.into() })
                } else {
                    Err(ProbError::BadProbability)
                }
            }
        }

        impl TryFrom<&$type> for Probability {
            type Error = ProbError;

            fn try_from(p: &$type) -> ProbResult<Self> {
                return (*p).try_into();
            }
        }
    }
);
impl_probability_try_from!(f64);
impl_probability_try_from!(f32);

// macro_rules! impl_probability_op(
//     ( $trt:ident, $op:tt, $fun:ident ) => {
//         impl $trt<Probability> for Probability {
//             type Output = f64;
//
//             fn $fun(self, rhs: Probability) -> f64 { self.p() $op rhs.p() }
//         }
//
//         impl $trt<f64> for Probability {
//             type Output = f64;
//
//             fn $fun(self, rhs: f64) -> f64 { self.p() $op rhs }
//         }
//     }
// );
// impl_probability_op!(Add, +, add);
// impl_probability_op!(Sub, -, sub);
// impl_probability_op!(Mul, *, mul);
// impl_probability_op!(Div, /, div);

impl Probability {
    /// Create a new `Probability`. `p` must be between zero and one
    /// (inclusive).
    pub fn new<P>(p: P) -> ProbResult<Self>
    where P: TryInto<Probability, Error = ProbError>
    {
        return p.try_into();
    }

    // pub fn new_unchecked(p: f64) -> Self { Probability { p } }

    pub fn random_rng<R>(rng: &mut R) -> Self
    where R: Rng + ?Sized
    {
        return Probability { p: rng.gen::<f64>() };
    }

    /// Generate a `Probability` uniformly sampled over $`[0, 1)`$
    pub fn random() -> Self {
        let mut rng = rnd::thread_rng();
        return Self::random_rng(&mut rng);
    }

    /// Create an `Iterator` for which `next` always returns a randomly
    /// generated `Probability`.
    pub fn generator() -> RandomProbGen {
        return RandomProbGen { rng: rnd::thread_rng() };
    }

    /// Access the stored numerical value.
    pub fn p(&self) -> f64 { self.p }

    /// Return `true` if `self.p() <= thresh`, `false` otherwise.
    pub fn ok_unchecked(&self, thresh: f64) -> bool { self.p <= thresh }
    
    /// Like `ok_unchecked`, but first checks that `thresh` is a valid
    /// probability.
    pub fn ok<P>(&self, thresh: P) -> ProbResult<bool>
    where P: TryInto<Probability, Error = ProbError>
    {
        let thresh_prob: Probability = thresh.try_into()?;
        return Ok(self.ok_unchecked(thresh_prob.p()));
    }
}

impl fmt::Display for Probability {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        return write!(f, "P:{}", self.p());
    }
}

/// An infinite `Iterator` for generating `Probability`s
#[derive(Clone)]
pub struct RandomProbGen {
    rng: rnd::ThreadRng
}

impl Iterator for RandomProbGen {
    type Item = Probability;

    fn next(&mut self) -> Option<Probability> {
        let p: f64 = self.rng.gen();
        return Some(Probability { p });
    }
}

/// Provides methods for sampling a random discrete variable according to some
/// distribution.
pub trait RandomDiscrete {
    type Var;

    fn sample_rng<R>(&self, rng: &mut R) -> Self::Var
    where R: Rng + ?Sized;

    /// Sample a single value.
    fn sample(&self) -> Self::Var {
        let mut rng = rnd::thread_rng();
        return self.sample_rng(&mut rng);
    }

    /// Return an infinite `Iterator` that samples the distribution.
    fn sampler(&self) -> DiscreteSampler<Self> {
        return DiscreteSampler { var: self, rng: rnd::thread_rng() };
    }

    /// Sample `N` values from the distribution.
    fn sample_for(&self, N: usize) -> Vec<Self::Var> {
        return self.sampler().take(N).collect();
    }

    /// Sample values from the distribution until some condition is met.
    fn sample_until<F>(&self, mut cond: F) -> Vec<Self::Var>
    where F: FnMut(Self::Var) -> Option<Self::Var>
    {
        return self.sampler().map_while(|v| cond(v)).collect();
    }
}

/// An infinite `Iterator` for sampling a discrete distribution.
#[derive(Clone)]
pub struct DiscreteSampler<'a, V>
where V: RandomDiscrete + ?Sized
{
    var: &'a V,
    rng: rnd::ThreadRng,
}

impl<'a, V> Iterator for DiscreteSampler<'a, V>
where V: RandomDiscrete + ?Sized
{
    type Item = <V as RandomDiscrete>::Var;

    fn next(&mut self) -> Option<Self::Item> {
        return Some(self.var.sample_rng(&mut self.rng));
    }
}

/// A random discrete variable obeying some distribution.
///
/// Constructors truncate sets of data to satisfy normalization conditions. No
/// ordering over values is required; values are taken in the order in which
/// they are first seen.
#[derive(Clone, Debug)]
pub struct DiscreteVariable<V>
where V: Clone
{
    dist: Vec<(V, f64, f64)>
}

impl<V> DiscreteVariable<V>
where V: Clone
{
    /// Construct from a series of values associated with probabilities.
    pub fn new<I>(prob: I) -> Self
    where I: IntoIterator<Item = (V, f64)>
    {
        let mut norm: f64 = 0.0;
        let p: Vec<(V, f64, f64)>
            = prob.into_iter()
            .map_while(|(v, p)| {
                if p < 0.0 {
                    panic!("encountered negative probability");
                }
                if norm + p > 1.0 - f64::EPSILON {
                    None
                } else {
                    norm += p;
                    Some((v, p, norm))
                }
            })
            .collect();
        let dist: Vec<(V, f64, f64)>
            = p.into_iter()
            .map(|(v, pmf, cdf)| (v, pmf / norm, cdf / norm))
            .collect();
        return Self { dist };
    }

    /// Construct from an empirical set of data.
    pub fn from_data<I>(data: I) -> Self
    where
        I: IntoIterator<Item = V>,
        V: Eq + std::hash::Hash,
    {
        let mut norm: f64 = 0.0;
        let mut hist: HashMap<V, usize> = HashMap::new();
        for v in data.into_iter() {
            if let Some(count) = hist.get_mut(&v) {
                *count += 1;
            } else {
                hist.insert(v, 1);
            }
            norm += 1.0;
        }
        let mut cdf_acc: f64 = 0.0;
        let dist: Vec<(V, f64, f64)>
            = hist.drain()
            .map(|(v, n)| {
                cdf_acc += n as f64 / norm;
                (v, n as f64 / norm, cdf_acc)
            })
            .collect();
        return Self { dist };
    }

    /// Construct from a series of values covering some domain and a PMF defined
    /// over it.
    pub fn from_pmf<I, F>(domain_gen: I, pmf: F) -> Self
    where
        I: IntoIterator<Item = V>,
        F: Fn(&V) -> f64,
    {
        return Self::new(
            domain_gen.into_iter()
                .map(|v| { let p: f64 = pmf(&v); (v, p) })
        );
    }

    /// Construct from a series of values covering some domain and a CDF defined
    /// over it.
    pub fn from_cdf<I, F>(domain_gen: I, cdf: F) -> Self
    where
        I: IntoIterator<Item = V>,
        F: Fn(&V) -> f64,
    {
        let mut cdf_last: f64 = 0.0;
        let mut cdf_cur: f64 = 0.0;
        let mut pmf_cur: f64 = 0.0;
        let p: Vec<(V, f64, f64)>
            = domain_gen.into_iter()
            .map_while(|v| {
                cdf_cur = cdf(&v);
                if cdf_cur < cdf_last || cdf_cur < 0.0 {
                    panic!("cdf must be monotonically increasing from zero");
                }
                if cdf_cur > 1.0 - f64::EPSILON {
                    None
                } else {
                    pmf_cur = cdf_cur - cdf_last;
                    cdf_last = cdf_cur;
                    Some((v, pmf_cur, cdf_cur))
                }
            })
            .collect();
        let dist: Vec<(V, f64, f64)>
            = p.into_iter()
            .map(|(v, pmf, cdf)| (v, pmf / cdf_last, cdf / cdf_last))
            .collect();
        return Self { dist };
    }

    /// Return the values in the domain over which the usual probability rules
    /// are effectively satisfied; i.e. probabilities sum to within machine
    /// epsilon of one.
    pub fn get_eff_domain(&self) -> Vec<V> {
        return self.dist.iter().map(|(v, _, _)| v.clone()).collect();
    }

    /// Return values in the effective domain with their probabilities.
    pub fn get_eff_pmf(&self) -> Vec<(V, f64)> {
        return self.dist.iter().map(|(v, pmf, _)| (v.clone(), *pmf)).collect();
    }

    /// Return values in the effective domain with their cumulative
    /// probabilities.
    pub fn get_eff_cdf(&self) -> Vec<(V, f64)> {
        return self.dist.iter().map(|(v, _, cdf)| (v.clone(), *cdf)).collect();
    }

    /// Return values in the effective domain with their (cumulative)
    /// probabilities.
    pub fn get_eff_dist(&self) -> Vec<(V, f64, f64)> { self.dist.clone() }
}

impl<V> RandomDiscrete for DiscreteVariable<V>
where V: Clone
{
    type Var = V;

    fn sample_rng<R>(&self, rng: &mut R) -> V
    where R: Rng + ?Sized
    {
        let r: f64 = rng.gen();
        let mut ret: &V = &self.dist.first().unwrap().0;
        for (v, _, f) in self.dist.iter() {
            if r < *f {
                ret = v;
                break;
            }
        }
        return ret.clone();
    }
}

/// Provides methods for sampling a random continuous variable according to some
/// distribution.
pub trait RandomContinuous {
    type Var: Float;

    fn sample_rng<R>(&self, rng: &mut R) -> Self::Var
    where R: Rng + ?Sized;

    /// Sample a single value.
    fn sample(&self) -> Self::Var {
        let mut rng = rnd::thread_rng();
        return self.sample_rng(&mut rng);
    }

    /// Return an finite `Iterator` that samples the distribution.
    fn sampler(&self) -> ContinuousSampler<Self> {
        return ContinuousSampler { var: self, rng: rnd::thread_rng() };
    }

    /// Sample `N` values from the distribution.
    fn sample_for(&self, N: usize) -> Vec<Self::Var> {
        return self.sampler().take(N).collect();
    }

    /// Sample values from the distribution until some condition is met.
    fn sample_until<F>(&self, mut cond: F) -> Vec<Self::Var>
    where F: FnMut(Self::Var) -> Option<Self::Var>
    {
        return self.sampler().map_while(|v| cond(v)).collect();
    }
}

/// An infinite `Iterator` for sampling a continuous distribution
#[derive(Clone)]
pub struct ContinuousSampler<'a, V>
where V: RandomContinuous + ?Sized
{
    var: &'a V,
    rng: rnd::ThreadRng,
}

impl<'a, V> Iterator for ContinuousSampler<'a, V>
where V: RandomContinuous + ?Sized
{
    type Item = <V as RandomContinuous>::Var;

    fn next(&mut self) -> Option<Self::Item> {
        return Some(self.var.sample_rng(&mut self.rng));
    }
}

/// A random continuous variable obeying some distribution.
///
/// Constructors truncate sets of data to satisfy normalization conditions.
/// Values of the independent variable must be seen by the constructor in
/// increasing order. Linear interpolation is used for sampling.
#[derive(Clone, Debug)]
pub struct ContinuousVariable<V>
where
    V: Float + Mul<f64, Output = V>,
    f64: Mul<V, Output = f64>,
{
    dist: Vec<(V, f64, f64)>
}

impl<V> ContinuousVariable<V>
where
    V: Float + Mul<f64, Output = V>,
    f64: Mul<V, Output = f64>,
{
    /// Construct from a series of values associated with probability densities.
    pub fn new<I>(prob_dens: I) -> Self
    where I: IntoIterator<Item = (V, f64)>
    {
        // prob_dens could be infinite -- take only what we need by integrating
        // via nonuniform trapezoidal until acc == 1.0
        // need to track two initialization conditions: one for (x, p) and one
        // for dx -- not fully initialized until the third item of prob_dens
        let mut x_last: V = <V as Float>::nan();
        let mut pdf_last: f64 = f64::NAN;
        let mut first: bool = true;
        let mut dx_cur: V = <V as Float>::nan();
        let mut dx_last: V = <V as Float>::nan();
        let mut dx_uninit: bool = true;
        let mut dcdf: f64 = f64::NAN;
        let mut even_spacing: bool = true;
        let mut acc: f64 = 0.0;
        let (x, (pdf, cdf)): (Vec<V>, (Vec<f64>, Vec<f64>))
            = prob_dens.into_iter()
            .map_while(|(v, p)| {
                if v < x_last {
                    panic!(
                        "independent variable values must be in increasing\
                         order"
                    );
                }
                if p < 0.0 {
                    panic!("encountered negative probability density");
                }
                if dx_uninit {
                    if first {
                        x_last = v;
                        pdf_last = p;
                        first = false;
                        Some((v, (p, 0.0)))
                    } else {
                        dx_cur = v - x_last;
                        dx_uninit = false;
                        acc += (pdf_last + p) * (dx_cur * 0.5);
                        x_last = v;
                        pdf_last = p;
                        dx_last = dx_cur;
                        Some((v, (p, acc)))
                    }
                } else {
                    dx_cur = v - x_last;
                    dcdf = (pdf_last + p) * (dx_cur * 0.5);
                    if acc + dcdf > 1.0 - f64::EPSILON {
                        None
                    } else {
                        if even_spacing && dx_cur != dx_last {
                            even_spacing = false;
                        }
                        acc += dcdf;
                        x_last = v;
                        pdf_last = p;
                        dx_last = dx_cur;
                        Some((v, (p, acc)))
                    }
                }
            })
            .unzip();
        let x: nd::Array1<V> = nd::Array::from(x);
        let mut pdf: nd::Array1<f64> = nd::Array::from(pdf);
        let mut cdf: nd::Array1<f64> = nd::Array::from(cdf);

        // double-check the normalization with simpson's rule if feasible
        let n: usize = pdf.len();
        let norm: f64
            = if n % 2 == 0 {
                acc
            } else if even_spacing {
                integrate::simpson(&pdf, &(x[1] - x[0]))
            } else {
                acc
            };
        pdf.mapv_inplace(|p| p / norm);
        cdf.mapv_inplace(|c| c / norm);

        let dist: Vec<(V, f64, f64)>
            = x.into_iter().zip(pdf.into_iter().zip(cdf.into_iter()))
            .map(|(xk, (pdfk, cdfk))| (xk, pdfk, cdfk))
            .collect();
        return Self { dist };
    }

    /// Construct from a series of values covering some domain and a PDF defined
    /// over it.
    pub fn from_pdf<I, F>(domain_gen: I, pdf: F) -> Self
    where
        I: IntoIterator<Item = V>,
        F: Fn(&V) -> f64,
    {
        return Self::new(domain_gen.into_iter().map(|v| (v, pdf(&v))));
    }

    /// Construct from a series of values covering some domain and a CDF defined
    /// over it.
    pub fn from_cdf<I, F>(domain_gen: I, cdf: F) -> Self
    where
        I: IntoIterator<Item = V>,
        F: Fn(&V) -> f64,
    {
        // the pdf is calculated in such a way that the cdf will be reproducible
        // through trapezoial rule integration
        let mut x_last: V = <V as Float>::nan();
        let mut pdf_last: f64 = f64::NAN;
        let mut pdf_cur: f64 = f64::NAN;
        let mut cdf_last: f64 = f64::NAN;
        let mut cdf_cur: f64 = f64::NAN;
        let mut first: bool = true;
        let mut dx_cur: V = <V as Float>::nan();
        let mut dx_last: V = <V as Float>::nan();
        let mut dx_uninit: bool = true;
        let mut even_spacing: bool = true;
        let mut acc: f64 = 0.0;
        let (x, (mut pdf, cdf)): (Vec<V>, (Vec<f64>, Vec<f64>))
            = domain_gen.into_iter()
            .map_while(|v| {
                if v < x_last {
                    panic!(
                        "independent variable values must be in increasing\
                         order"
                    );
                }
                if dx_uninit {
                    if first {
                        x_last = v;
                        cdf_last = cdf(&v);
                        if cdf_last < 0.0 + f64::EPSILON {
                            panic!(
                                "cdf must be monotonically increasing and\
                                 positive"
                            );
                        }
                        first = false;
                        Some((v, (0.0, cdf_last)))
                    } else {
                        dx_cur = v - x_last;
                        dx_uninit = false;
                        cdf_cur = cdf(&v);
                        if cdf_cur < cdf_last || cdf_cur < 0.0 {
                            panic!(
                                "cdf must be monotonically increasing and\
                                 positive"
                            );
                        }
                        pdf_cur = (
                            (cdf_cur - cdf_last)
                                * (dx_cur.powf(-<V as One>::one()) * 2.0)
                            - pdf_last
                        );
                        acc += (pdf_last + pdf_cur) * (dx_cur * 0.5);
                        x_last = v;
                        pdf_last = pdf_cur;
                        cdf_last = cdf_cur;
                        dx_last = dx_cur;
                        Some((v, (pdf_cur, cdf_cur)))
                    }
                } else {
                    cdf_cur = cdf(&v);
                    if cdf_cur < cdf_last || cdf_cur < 0.0 {
                        panic!(
                            "cdf must be monotonically increasing and\
                             positive"
                        );
                    }
                    if cdf_cur > 1.0 - f64::EPSILON {
                        None
                    } else {
                        dx_cur = v - x_last;
                        if even_spacing && dx_cur != dx_last {
                            even_spacing = false;
                        }
                        pdf_cur = (
                            (cdf_cur - cdf_last)
                                * (dx_cur.powf(-<V as One>::one()) * 2.0)
                            - pdf_last
                        );
                        acc += (pdf_last + pdf_cur) * (dx_cur * 0.5);
                        x_last = v;
                        pdf_last = pdf_cur;
                        cdf_last = cdf_cur;
                        dx_last = dx_cur;
                        Some((v, (pdf_cur, cdf_cur)))
                    }
                }
            })
            .unzip();

        // fine-tune the normalization:
        // * pdf is off by a constant because we set the first value to zero
        // * cdf_last could be off from 1 by a small value
        let n: usize = x.len();
        let pdf_shift_tot: f64 = cdf_last - acc;
        let pdf_shift: Vec<f64>
            = x.iter().take(n - 1).zip(x.iter().skip(1))
            .map(|(xk, xkp1)| {
                pdf_shift_tot / (n - 1) as f64
                    * (*xkp1 - *xk).powf(-<V as One>::one())
            })
            .collect();
        pdf.iter_mut()
            .zip(
                Iterator::zip(
                    [0.0_f64].iter().chain(pdf_shift.iter()),
                    pdf_shift.iter().chain([0.0_f64].iter()),
                )
            )
            .for_each(|(p, (dpl, dpr))| { *p += *dpl + *dpr; });

        let dist: Vec<(V, f64, f64)>
            = x.into_iter().zip(pdf.into_iter().zip(cdf.into_iter()))
            .map(|(xk, (pdfk, cdfk))| (xk, pdfk / cdf_last, cdfk / cdf_last))
            .collect();
        return Self { dist };
    }

    /// Return the values in the domain over which the usual probability rules
    /// are effectively satisfied; i.e. probabilities sum to within machine
    /// epsilon of one.
    pub fn get_eff_domain(&self) -> Vec<V> {
        return self.dist.iter().map(|(v, _, _)| *v).collect();
    }

    /// Return values in the effective domain with their probability densities.
    pub fn get_eff_pdf(&self) -> Vec<(V, f64)> {
        return self.dist.iter().map(|(v, pdf, _)| (*v, *pdf)).collect();
    }

    /// Return values in the effective domain with their cumulative
    /// probabilities.
    pub fn get_eff_cdf(&self) -> Vec<(V, f64)> {
        return self.dist.iter().map(|(v, _, cdf)| (*v, *cdf)).collect();
    }

    /// Return values in the effective domain with their (cumulative)
    /// probabilities (densities).
    pub fn get_eff_dist(&self) -> Vec<(V, f64, f64)> { self.dist.clone() }
}

impl<V> RandomContinuous for ContinuousVariable<V>
where
    V: Float + Mul<f64, Output = V>,
    f64: Mul<V, Output = f64>,
{
    type Var = V;

    fn sample_rng<R>(&self, rng: &mut R) -> V
    where R: Rng + ?Sized
    {
        let r: f64 = rng.gen();
        let mut ret: V = <V as Float>::nan();
        let (mut v_last, _, mut f_last): (V, _, f64)
            = *self.dist.first().unwrap();
        for (v, _, f) in self.dist.iter() {
            if r < *f {
                ret = (
                    *v
                    + ((*v - v_last) * (*f - f_last).powi(-1))
                        * (r - f_last)
                );
                break;
            }
            v_last = *v;
            f_last = *f;
        }
        return ret;
    }
}

/// Single sum type describing all kinds of (semi-)definite and (semi-)infinite
/// ranges of floating-point numbers describable by Rust's native `..` syntax.
#[derive(Clone, Copy, Debug)]
pub enum FloatRange<V>
where V: Float
{
    /// Represents a range of the form $`[a, \infty)`$
    From(V),

    /// Represents a range of the form $`[a, b)`$
    FromTo(V, V),

    /// Represents a range of the form $`[a, b]`$
    FromToInc(V, V),

    /// Represents a range of the form $`(-\infty, b)`$
    To(V),

    /// Represents a range of the form $`(-\infty, b]`$
    ToInc(V),

    /// Represents a range of the form $`(-\infty, +\infty)`$
    Unbounded,
}

impl<V> From<RangeFrom<V>> for FloatRange<V>
where V: Float
{
    fn from(range: RangeFrom<V>) -> Self {
        return FloatRange::From(range.start);
    }
}

impl<V> From<Range<V>> for FloatRange<V>
where V: Float
{
    fn from(range: Range<V>) -> Self {
        return FloatRange::FromTo(range.start, range.end);
    }
}

impl<V> From<RangeInclusive<V>> for FloatRange<V>
where V: Float
{
    fn from(range: RangeInclusive<V>) -> Self {
        return FloatRange::FromToInc(*range.start(), *range.end());
    }
}

impl<V> From<RangeTo<V>> for FloatRange<V>
where V: Float
{
    fn from(range: RangeTo<V>) -> Self {
        return FloatRange::To(range.end);
    }
}

impl<V> From<RangeToInclusive<V>> for FloatRange<V>
where V: Float
{
    fn from(range: RangeToInclusive<V>) -> Self {
        return FloatRange::ToInc(range.end);
    }
}

impl<V> From<RangeFull> for FloatRange<V>
where V: Float
{
    fn from(_range: RangeFull) -> Self {
        return FloatRange::Unbounded;
    }
}

impl<V> Region for FloatRange<V>
where V: Float
{
    type Domain = V;

    fn contains(&self, x: &V) -> bool {
        return match self {
            FloatRange::From(s) => x >= s,
            FloatRange::FromTo(s, e) => x >= s && x < e,
            FloatRange::FromToInc(s, e) => x >= s && x <= e,
            FloatRange::To(e) => x < e,
            FloatRange::ToInc(e) => x <= e,
            FloatRange::Unbounded => true,
        };
    }

    fn clamp(&self, x: V) -> V {
        return if let FloatRange::From(s) = self {
            if x < *s { *s }
            else { x }
        } else if let FloatRange::FromTo(s, e) = self {
            if x < *s { *s }
            else if x >= *e { *e - <V as Float>::epsilon() }
            else { x }
        } else if let FloatRange::FromToInc(s, e) = self {
            if x < *s { *s }
            else if x > *e { *e }
            else { x }
        } else if let FloatRange::To(e) = self {
            if x >= *e { *e - <V as Float>::epsilon() }
            else { x }
        } else if let FloatRange::ToInc(e) = self {
            if x > *e { *e }
            else { x }
        } else {
            x
        };
    }
}

/// A random continuous variable obeying some distribution described by
/// functional form.
///
/// This type is very fragile in the sense that the PDF and CDF provided by the
/// user are not verified as such and samples are generated via simple
/// Newton-Raphson. Because of this, methods are very prone to pathologies and
/// can fail unexpectedly.
///
/// Construction requires a definition of the domain over which the PDF and CDF
/// are defined as well as some point (any point, preferrably somewhere close to
/// the middle of the distribution) within it for Newton-Raphson.
#[derive(Clone, Debug)]
pub struct ContinuousVariableFunc<V, PDF, CDF>
where
    V: Float + SubAssign<f64> + Epsilon,
    PDF: Fn(&V) -> f64 + Clone,
    CDF: Fn(&V) -> f64 + Clone,
{
    domain: FloatRange<V>,
    inside: V,
    pdf_func: PDF,
    cdf_func: CDF,
    nr_options: NROptions,
}

impl<V, PDF, CDF> ContinuousVariableFunc<V, PDF, CDF>
where
    V: Float + SubAssign<f64> + Epsilon,
    PDF: Fn(&V) -> f64 + Clone,
    CDF: Fn(&V) -> f64 + Clone,
{
    /// Construct from a domain of floating-point values and a PDF and CDF
    /// defined over it. A single point `inside` within the domain is required
    /// for sampling via Newton-Raphson. Parameters for the Newton-Raphson
    /// routine can be supplied in `nr_options` (see [NROptions]).
    pub fn new<D>(
        domain: D,
        inside: V,
        pdf: PDF,
        cdf: CDF,
        nr_options: Option<NROptions>
    ) -> Self
    where D: Into<FloatRange<V>>
    {
        return Self {
            domain: domain.into(),
            inside,
            pdf_func: pdf,
            cdf_func: cdf,
            nr_options: nr_options.unwrap_or(NROptions::default()),
        };
    }

    /// Pass-through method to access the PDF.
    pub fn pdf(&self, x: &V) -> f64 { (self.pdf_func)(x) }

    /// Pass-through method to access the CDF.
    pub fn cdf(&self, x: &V) -> f64 { (self.cdf_func)(x) }
}

impl<V, PDF, CDF> RandomContinuous for ContinuousVariableFunc<V, PDF, CDF>
where
    V: Float + SubAssign<f64> + Epsilon,
    PDF: Fn(&V) -> f64 + Clone,
    CDF: Fn(&V) -> f64 + Clone,
{
    type Var = V;

    fn sample_rng<R>(&self, rng: &mut R) -> V
    where R: Rng + ?Sized
    {
        let r: f64 = rng.gen();
        return search::find_root_1d(
            self.inside,
            |x: &V| -> f64 { (self.cdf_func)(x) - r },
            self.pdf_func.clone(),
            self.domain,
            self.nr_options.epsilon,
            self.nr_options.maxiters,
        );
    }
}

