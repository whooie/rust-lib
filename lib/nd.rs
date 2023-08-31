//! Provides the `ND` enum, which wraps values that can be operated on within
//! the `ndarray` ecosystem.
//!
//! Specifically, it's meant to hold either a scalar `A` or an array
//! `nd::Array<A, D>` and seamlessly handle method calls and mathematical
//! interactions between them and itself.

use std::{
    ops::{
        // Neg,
        Add,
        Sub,
        Mul,
        Div,
    },
};
use ndarray::{
    self as nd,
    Data,
    DataMut,
    OwnedRepr,
    ViewRepr,
    Dimension,
    ScalarOperand,
    Ix1,
    Ix2,
    Ix3,
    Ix4,
    Ix5,
    Ix6,
    IxDyn,
};
use crate::mkerr;

mkerr!(
    NDError : {
        NDIntoScalar => "not a scalar",
        NDIntoArray => "not an array",
        ConvertDimensionality => "could not convert dimensionality",
    }
);
pub type NDResult<T> = Result<T, NDError>;

/// Base `ND` type.
///
/// Stored values must be able to be held and operated on by `ndarray`
/// constructs. `ND`s holding built-in numerical types implement "pass-through"
/// methods that operate on scalar variants identically to the unwrapped value,
/// or element-wise on array variants.
///
/// Also implement the standard arithmetic traits `Add`, ..., `Div` giving
/// limited interop with bare arrays and inheriting the need for borrowing
/// values in these operations from the `ndarray::ArrayBase` type.
#[derive(Debug, PartialEq)]
pub enum NDBase<T, S, D>
where S: Data<Elem=T>,
      D: Dimension,
{
    S(T),
    A(nd::ArrayBase<S, D>),
}

impl<T, D> Clone for NDBase<T, OwnedRepr<T>, D>
where T: Clone,
      D: Dimension,
{
    fn clone(&self) -> Self {
        return match self {
            NDBase::S(s) => NDBase::S(s.clone()),
            NDBase::A(a) => NDBase::A(a.clone()),
        };
    }
}

impl<'a, T, D> Clone for NDBase<T, ViewRepr<&'a T>, D>
where T: Clone,
      D: Dimension,
{
    fn clone(&self) -> Self {
        return match self {
            NDBase::S(s) => NDBase::S(s.clone()),
            NDBase::A(a) => NDBase::A(a.clone()),
        };
    }
}

impl<'a, T, S, D> NDBase<T, S, D>
where T: Clone,
      S: Data<Elem=T>,
      D: Dimension,
{
    pub fn view(&'a self) -> NDBase<T, ViewRepr<&'a T>, D> {
        return match self {
            NDBase::S(s) => NDBase::S(s.clone()),
            NDBase::A(a) => NDBase::A(a.view()),
        };
    }
}

macro_rules! impl_nd_from_t(
    ( $sh:ident ) => {
        impl<T> From<T> for NDBase<T, OwnedRepr<T>, $sh> {
            fn from(x: T) -> NDBase<T, OwnedRepr<T>, $sh> { NDBase::S(x) }
        }
    }
);

impl_nd_from_t!(Ix1);
impl_nd_from_t!(Ix2);
impl_nd_from_t!(Ix3);
impl_nd_from_t!(Ix4);
impl_nd_from_t!(Ix5);
impl_nd_from_t!(Ix6);

impl<T, S, D> From<nd::ArrayBase<S, D>> for NDBase<T, S, D>
where S: Data<Elem=T>,
      D: Dimension,
{
    fn from(x: nd::ArrayBase<S, D>) -> NDBase<T, S, D> { NDBase::A(x) }
}

impl<T, S, D> Default for NDBase<T, S, D>
where T: Default,
      S: Data<Elem=T>,
      D: Dimension,
{
    fn default() -> Self { NDBase::S(T::default()) }
}

impl<T, S, D> NDBase<T, S, D>
where S: Data<Elem=T>,
      D: Dimension,
{
    pub fn is_scalar(&self) -> bool {
        return match self {
            NDBase::S(_) => true,
            NDBase::A(_) => false,
        };
    }

    pub fn is_array(&self) -> bool {
        return match self {
            NDBase::S(_) => false,
            NDBase::A(_) => true,
        };
    }

    pub fn get_scalar(&self) -> Option<&T> {
        return match self {
            NDBase::S(s) => Some(s),
            NDBase::A(_) => None,
        };
    }

    pub fn get_array(&self) -> Option<&nd::ArrayBase<S, D>> {
        return match self {
            NDBase::S(_) => None,
            NDBase::A(a) => Some(a),
        };
    }

    pub fn get_array_view(&self) -> Option<nd::ArrayView<T, D>> {
        return match self {
            NDBase::S(_) => None,
            NDBase::A(a) => Some(a.view()),
        };
    }

    pub fn into_scalar(self) -> NDResult<T> {
        return match self {
            NDBase::S(s) => Ok(s),
            NDBase::A(_) => Err(NDError::NDIntoScalar),
        };
    }

    pub fn into_array(self) -> NDResult<nd::ArrayBase<S, D>> {
        return match self {
            NDBase::S(_) => Err(NDError::NDIntoArray),
            NDBase::A(a) => Ok(a),
        };
    }

    pub fn into_dimensionality<D2>(self) -> NDResult<NDBase<T, S, D2>>
    where D2: Dimension
    {
        return match self {
            NDBase::S(s) => Ok(NDBase::S(s)),
            NDBase::A(a) => {
                a.into_dimensionality::<D2>()
                    .map(|b| NDBase::A(b))
                    .map_err(|_| NDError::ConvertDimensionality)
            },
        };
    }

    pub fn map<'a, U, F>(&'a self, mut f: F) -> NDBase<U, OwnedRepr<U>, D>
    where F: FnMut(&'a T) -> U,
          T: 'a,
          NDBase<U, OwnedRepr<U>, D>: From<U> + From<nd::Array<U, D>>,
    {
        return match self {
            NDBase::S(s) => NDBase::<U, OwnedRepr<U>, D>::from(f(s)),
            NDBase::A(a) => NDBase::<U, OwnedRepr<U>, D>::from(a.map(f)),
        };
    }

    pub fn mapv<U, F>(&self, mut f: F) -> NDBase<U, OwnedRepr<U>, D>
    where F: FnMut(T) -> U,
          T: Clone,
          NDBase<U, OwnedRepr<U>, D>: From<U> + From<nd::Array<U, D>>,
    {
        return match self {
            NDBase::S(s) => NDBase::<U, OwnedRepr<U>, D>::from(f(s.clone())),
            NDBase::A(a) => NDBase::<U, OwnedRepr<U>, D>::from(a.mapv(f)),
        };
    }

    pub fn for_each<'a, F>(&'a self, mut f: F)
    where F: FnMut(&'a T),
          T: 'a,
    {
        match self {
            NDBase::S(s) => { f(s); },
            NDBase::A(a) => { a.for_each(f); },
        };
    }
}

impl<T, S, D> NDBase<T, S, D>
where S: Data<Elem=T> + DataMut,
      D: Dimension,
{
    pub fn map_inplace<'a, F>(&'a mut self, mut f: F)
    where F: FnMut(&'a mut T),
          T: 'a,
    {
        match self {
            NDBase::S(s) => { f(s); },
            NDBase::A(a) => { a.map_inplace(f); },
        }
    }

    pub fn mapv_inplace<F>(&mut self, mut f: F)
    where F: FnMut(T) -> T,
          T: Clone,
    {
        match self {
            NDBase::S(s) => { *s = f(s.clone()); },
            NDBase::A(a) => { a.mapv_inplace(f); },
        }
    }

    pub fn mapv_into<F>(mut self, f: F) -> Self
    where F: FnMut(T) -> T,
          T: Clone,
    {
        self.mapv_inplace(f);
        return self;
    }
}

impl<T, S, D> NDBase<T, S, D>
where T: Clone,
      S: Data<Elem=T>,
      D: Dimension,
      nd::ArrayBase<S, D>: Clone,
{
    pub fn get_cloned_scalar(&self) -> Option<T> {
        return match self {
            NDBase::S(s) => Some(s.clone()),
            NDBase::A(_) => None,
        };
    }

    pub fn get_cloned_array(&self) -> Option<nd::ArrayBase<S, D>> {
        return match self {
            NDBase::S(_) => None,
            NDBase::A(a) => Some(a.clone()),
        };
    }

    pub fn to_owned(&self) -> NDBase<T, OwnedRepr<T>, D> {
        return match self {
            NDBase::S(s) => NDBase::S(s.clone()),
            NDBase::A(a) => NDBase::A(a.to_owned()),
        };
    }

    pub fn to_dimensionality<D2>(&self) -> NDResult<NDBase<T, S, D2>>
    where D2: Dimension
    {
        return match self {
            NDBase::S(s) => Ok(NDBase::S(s.clone())),
            NDBase::A(a) => {
                a.clone().into_dimensionality::<D2>()
                    .map(|b| NDBase::A(b))
                    .map_err(|_| NDError::ConvertDimensionality)
            },
        };
    }
}

macro_rules! impl_nd_passthrough_methods(
    (
        $type:ty : { $(
            $meth:ident ( $( $arg:ident : $argtype:ty ),* ) -> $rettype:ty
        ),+ $(,)? }
    ) => {
        impl<S, D> NDBase<$type, S, D>
        where S: Data<Elem=$type>,
              D: Dimension,
        { $(
            pub fn $meth(&self $(, $arg: $argtype )*)
                -> NDBase<$rettype, OwnedRepr<$rettype>, D>
            {
                return match self {
                    NDBase::S(s)
                        => NDBase::S(s.$meth( $( $arg ),* )),
                    NDBase::A(a)
                        => NDBase::A(
                            a.mapv(
                                |ak| <$type>::$meth(ak $(, $arg )* ))
                        ),
                };
            }
        )+ }
    }
);

impl_nd_passthrough_methods!(
    f64 : {
        floor ( ) -> f64,
        ceil ( ) -> f64,
        round ( ) -> f64,
        trunc ( ) -> f64,
        fract ( ) -> f64,
        abs ( ) -> f64,
        signum ( ) -> f64,
        copysign ( sign: f64 ) -> f64,
        mul_add ( a: f64, b: f64 ) -> f64,
        div_euclid ( rhs: f64 ) -> f64,
        rem_euclid ( rhs: f64 ) -> f64,
        powi ( n: i32 ) -> f64,
        powf ( n: f64 ) -> f64,
        sqrt ( ) -> f64,
        exp ( ) -> f64,
        exp2 ( ) -> f64,
        ln ( ) -> f64,
        log ( base: f64 ) -> f64,
        log2 ( ) -> f64,
        log10 ( ) -> f64,
        cbrt ( ) -> f64,
        hypot ( other: f64 ) -> f64,
        sin ( ) -> f64,
        cos ( ) -> f64,
        tan ( ) -> f64,
        asin ( ) -> f64,
        acos ( ) -> f64,
        atan ( ) -> f64,
        atan2 ( other: f64 ) -> f64,
        exp_m1 ( ) -> f64,
        ln_1p ( ) -> f64,
        sinh ( ) -> f64,
        cosh ( ) -> f64,
        tanh ( ) -> f64,
        asinh ( ) -> f64,
        acosh ( ) -> f64,
        atanh ( ) -> f64,
        is_nan ( ) -> bool,
        is_infinite ( ) -> bool,
        is_finite ( ) -> bool,
        is_subnormal ( ) -> bool,
        is_normal ( ) -> bool,
        is_sign_positive ( ) -> bool,
        is_sign_negative ( ) -> bool,
        recip ( ) -> f64,
        to_degrees ( ) -> f64,
        to_radians ( ) -> f64,
        max ( other: f64 ) -> f64,
        min ( other: f64 ) -> f64,
        to_bits ( ) -> u64,
        to_be_bytes ( ) -> [u8; 8],
        to_le_bytes ( ) -> [u8; 8],
        to_ne_bytes ( ) -> [u8; 8],
        clamp ( min: f64, max: f64 ) -> f64,
    }
);

impl_nd_passthrough_methods!(
    f32 : {
        floor ( ) -> f32,
        ceil ( ) -> f32,
        round ( ) -> f32,
        trunc ( ) -> f32,
        fract ( ) -> f32,
        abs ( ) -> f32,
        signum ( ) -> f32,
        copysign ( sign: f32 ) -> f32,
        mul_add ( a: f32, b: f32 ) -> f32,
        div_euclid ( rhs: f32 ) -> f32,
        rem_euclid ( rhs: f32 ) -> f32,
        powi ( n: i32 ) -> f32,
        powf ( n: f32 ) -> f32,
        sqrt ( ) -> f32,
        exp ( ) -> f32,
        exp2 ( ) -> f32,
        ln ( ) -> f32,
        log ( base: f32 ) -> f32,
        log2 ( ) -> f32,
        log10 ( ) -> f32,
        cbrt ( ) -> f32,
        hypot ( other: f32 ) -> f32,
        sin ( ) -> f32,
        cos ( ) -> f32,
        tan ( ) -> f32,
        asin ( ) -> f32,
        acos ( ) -> f32,
        atan ( ) -> f32,
        atan2 ( other: f32 ) -> f32,
        exp_m1 ( ) -> f32,
        ln_1p ( ) -> f32,
        sinh ( ) -> f32,
        cosh ( ) -> f32,
        tanh ( ) -> f32,
        asinh ( ) -> f32,
        acosh ( ) -> f32,
        atanh ( ) -> f32,
        is_nan ( ) -> bool,
        is_infinite ( ) -> bool,
        is_finite ( ) -> bool,
        is_subnormal ( ) -> bool,
        is_normal ( ) -> bool,
        is_sign_positive ( ) -> bool,
        is_sign_negative ( ) -> bool,
        recip ( ) -> f32,
        to_degrees ( ) -> f32,
        to_radians ( ) -> f32,
        max ( other: f32 ) -> f32,
        min ( other: f32 ) -> f32,
        to_bits ( ) -> u32,
        to_be_bytes ( ) -> [u8; 4],
        to_le_bytes ( ) -> [u8; 4],
        to_ne_bytes ( ) -> [u8; 4],
        clamp ( min: f32, max: f32 ) -> f32,
    }
);

impl_nd_passthrough_methods!(
    i128 : {
        count_ones ( ) -> u32,
        count_zeros ( ) -> u32,
        leading_zeros ( ) -> u32,
        trailing_zeros ( ) -> u32,
        leading_ones ( ) -> u32,
        trailing_ones ( ) -> u32,
        rotate_left ( n: u32 ) -> i128,
        rotate_right ( n: u32 ) -> i128,
        swap_bytes ( ) -> i128,
        reverse_bits ( ) -> i128,
        to_be ( ) -> i128,
        to_le ( ) -> i128,
        saturating_add ( rhs: i128 ) -> i128,
        saturating_sub ( rhs: i128 ) -> i128,
        saturating_neg ( ) -> i128,
        saturating_abs ( ) -> i128,
        saturating_mul ( rhs: i128 ) -> i128,
        saturating_div ( rhs: i128 ) -> i128,
        saturating_pow ( exp: u32 ) -> i128,
        wrapping_add ( rhs: i128 ) -> i128,
        wrapping_sub ( rhs: i128 ) -> i128,
        wrapping_mul ( rhs: i128 ) -> i128,
        wrapping_div ( rhs: i128 ) -> i128,
        wrapping_div_euclid ( rhs: i128 ) -> i128,
        wrapping_rem ( rhs: i128 ) -> i128,
        wrapping_rem_euclid ( rhs: i128 ) -> i128,
        wrapping_neg ( ) -> i128,
        wrapping_shl ( rhs: u32 ) -> i128,
        wrapping_shr ( rhs: u32 ) -> i128,
        wrapping_abs ( ) -> i128,
        unsigned_abs ( ) -> u128,
        wrapping_pow ( exp: u32 ) -> i128,
        pow ( exp: u32 ) -> i128,
        div_euclid ( rhs: i128 ) -> i128,
        rem_euclid ( rhs: i128 ) -> i128,
        abs ( ) -> i128,
        signum ( ) -> i128,
        is_positive ( ) -> bool,
        is_negative ( ) -> bool,
        to_be_bytes ( ) -> [u8; 16],
        to_le_bytes ( ) -> [u8; 16],
        to_ne_bytes ( ) -> [u8; 16],
    }
);

impl_nd_passthrough_methods!(
    i64 : {
        count_ones ( ) -> u32,
        count_zeros ( ) -> u32,
        leading_zeros ( ) -> u32,
        trailing_zeros ( ) -> u32,
        leading_ones ( ) -> u32,
        trailing_ones ( ) -> u32,
        rotate_left ( n: u32 ) -> i64,
        rotate_right ( n: u32 ) -> i64,
        swap_bytes ( ) -> i64,
        reverse_bits ( ) -> i64,
        to_be ( ) -> i64,
        to_le ( ) -> i64,
        saturating_add ( rhs: i64 ) -> i64,
        saturating_sub ( rhs: i64 ) -> i64,
        saturating_neg ( ) -> i64,
        saturating_abs ( ) -> i64,
        saturating_mul ( rhs: i64 ) -> i64,
        saturating_div ( rhs: i64 ) -> i64,
        saturating_pow ( exp: u32 ) -> i64,
        wrapping_add ( rhs: i64 ) -> i64,
        wrapping_sub ( rhs: i64 ) -> i64,
        wrapping_mul ( rhs: i64 ) -> i64,
        wrapping_div ( rhs: i64 ) -> i64,
        wrapping_div_euclid ( rhs: i64 ) -> i64,
        wrapping_rem ( rhs: i64 ) -> i64,
        wrapping_rem_euclid ( rhs: i64 ) -> i64,
        wrapping_neg ( ) -> i64,
        wrapping_shl ( rhs: u32 ) -> i64,
        wrapping_shr ( rhs: u32 ) -> i64,
        wrapping_abs ( ) -> i64,
        unsigned_abs ( ) -> u64,
        wrapping_pow ( exp: u32 ) -> i64,
        pow ( exp: u32 ) -> i64,
        div_euclid ( rhs: i64 ) -> i64,
        rem_euclid ( rhs: i64 ) -> i64,
        abs ( ) -> i64,
        signum ( ) -> i64,
        is_positive ( ) -> bool,
        is_negative ( ) -> bool,
        to_be_bytes ( ) -> [u8; 8],
        to_le_bytes ( ) -> [u8; 8],
        to_ne_bytes ( ) -> [u8; 8],
    }
);

impl_nd_passthrough_methods!(
    i32 : {
        count_ones ( ) -> u32,
        count_zeros ( ) -> u32,
        leading_zeros ( ) -> u32,
        trailing_zeros ( ) -> u32,
        leading_ones ( ) -> u32,
        trailing_ones ( ) -> u32,
        rotate_left ( n: u32 ) -> i32,
        rotate_right ( n: u32 ) -> i32,
        swap_bytes ( ) -> i32,
        reverse_bits ( ) -> i32,
        to_be ( ) -> i32,
        to_le ( ) -> i32,
        saturating_add ( rhs: i32 ) -> i32,
        saturating_sub ( rhs: i32 ) -> i32,
        saturating_neg ( ) -> i32,
        saturating_abs ( ) -> i32,
        saturating_mul ( rhs: i32 ) -> i32,
        saturating_div ( rhs: i32 ) -> i32,
        saturating_pow ( exp: u32 ) -> i32,
        wrapping_add ( rhs: i32 ) -> i32,
        wrapping_sub ( rhs: i32 ) -> i32,
        wrapping_mul ( rhs: i32 ) -> i32,
        wrapping_div ( rhs: i32 ) -> i32,
        wrapping_div_euclid ( rhs: i32 ) -> i32,
        wrapping_rem ( rhs: i32 ) -> i32,
        wrapping_rem_euclid ( rhs: i32 ) -> i32,
        wrapping_neg ( ) -> i32,
        wrapping_shl ( rhs: u32 ) -> i32,
        wrapping_shr ( rhs: u32 ) -> i32,
        wrapping_abs ( ) -> i32,
        unsigned_abs ( ) -> u32,
        wrapping_pow ( exp: u32 ) -> i32,
        pow ( exp: u32 ) -> i32,
        div_euclid ( rhs: i32 ) -> i32,
        rem_euclid ( rhs: i32 ) -> i32,
        abs ( ) -> i32,
        signum ( ) -> i32,
        is_positive ( ) -> bool,
        is_negative ( ) -> bool,
        to_be_bytes ( ) -> [u8; 4],
        to_le_bytes ( ) -> [u8; 4],
        to_ne_bytes ( ) -> [u8; 4],
    }
);

impl_nd_passthrough_methods!(
    i16 : {
        count_ones ( ) -> u32,
        count_zeros ( ) -> u32,
        leading_zeros ( ) -> u32,
        trailing_zeros ( ) -> u32,
        leading_ones ( ) -> u32,
        trailing_ones ( ) -> u32,
        rotate_left ( n: u32 ) -> i16,
        rotate_right ( n: u32 ) -> i16,
        swap_bytes ( ) -> i16,
        reverse_bits ( ) -> i16,
        to_be ( ) -> i16,
        to_le ( ) -> i16,
        saturating_add ( rhs: i16 ) -> i16,
        saturating_sub ( rhs: i16 ) -> i16,
        saturating_neg ( ) -> i16,
        saturating_abs ( ) -> i16,
        saturating_mul ( rhs: i16 ) -> i16,
        saturating_div ( rhs: i16 ) -> i16,
        saturating_pow ( exp: u32 ) -> i16,
        wrapping_add ( rhs: i16 ) -> i16,
        wrapping_sub ( rhs: i16 ) -> i16,
        wrapping_mul ( rhs: i16 ) -> i16,
        wrapping_div ( rhs: i16 ) -> i16,
        wrapping_div_euclid ( rhs: i16 ) -> i16,
        wrapping_rem ( rhs: i16 ) -> i16,
        wrapping_rem_euclid ( rhs: i16 ) -> i16,
        wrapping_neg ( ) -> i16,
        wrapping_shl ( rhs: u32 ) -> i16,
        wrapping_shr ( rhs: u32 ) -> i16,
        wrapping_abs ( ) -> i16,
        unsigned_abs ( ) -> u16,
        wrapping_pow ( exp: u32 ) -> i16,
        pow ( exp: u32 ) -> i16,
        div_euclid ( rhs: i16 ) -> i16,
        rem_euclid ( rhs: i16 ) -> i16,
        abs ( ) -> i16,
        signum ( ) -> i16,
        is_positive ( ) -> bool,
        is_negative ( ) -> bool,
        to_be_bytes ( ) -> [u8; 2],
        to_le_bytes ( ) -> [u8; 2],
        to_ne_bytes ( ) -> [u8; 2],
    }
);

impl_nd_passthrough_methods!(
    i8 : {
        count_ones ( ) -> u32,
        count_zeros ( ) -> u32,
        leading_zeros ( ) -> u32,
        trailing_zeros ( ) -> u32,
        leading_ones ( ) -> u32,
        trailing_ones ( ) -> u32,
        rotate_left ( n: u32 ) -> i8,
        rotate_right ( n: u32 ) -> i8,
        swap_bytes ( ) -> i8,
        reverse_bits ( ) -> i8,
        to_be ( ) -> i8,
        to_le ( ) -> i8,
        saturating_add ( rhs: i8 ) -> i8,
        saturating_sub ( rhs: i8 ) -> i8,
        saturating_neg ( ) -> i8,
        saturating_abs ( ) -> i8,
        saturating_mul ( rhs: i8 ) -> i8,
        saturating_div ( rhs: i8 ) -> i8,
        saturating_pow ( exp: u32 ) -> i8,
        wrapping_add ( rhs: i8 ) -> i8,
        wrapping_sub ( rhs: i8 ) -> i8,
        wrapping_mul ( rhs: i8 ) -> i8,
        wrapping_div ( rhs: i8 ) -> i8,
        wrapping_div_euclid ( rhs: i8 ) -> i8,
        wrapping_rem ( rhs: i8 ) -> i8,
        wrapping_rem_euclid ( rhs: i8 ) -> i8,
        wrapping_neg ( ) -> i8,
        wrapping_shl ( rhs: u32 ) -> i8,
        wrapping_shr ( rhs: u32 ) -> i8,
        wrapping_abs ( ) -> i8,
        unsigned_abs ( ) -> u8,
        wrapping_pow ( exp: u32 ) -> i8,
        pow ( exp: u32 ) -> i8,
        div_euclid ( rhs: i8 ) -> i8,
        rem_euclid ( rhs: i8 ) -> i8,
        abs ( ) -> i8,
        signum ( ) -> i8,
        is_positive ( ) -> bool,
        is_negative ( ) -> bool,
        to_be_bytes ( ) -> [u8; 1],
        to_le_bytes ( ) -> [u8; 1],
        to_ne_bytes ( ) -> [u8; 1],
    }
);

impl_nd_passthrough_methods!(
    isize : {
        count_ones ( ) -> u32,
        count_zeros ( ) -> u32,
        leading_zeros ( ) -> u32,
        trailing_zeros ( ) -> u32,
        leading_ones ( ) -> u32,
        trailing_ones ( ) -> u32,
        rotate_left ( n: u32 ) -> isize,
        rotate_right ( n: u32 ) -> isize,
        swap_bytes ( ) -> isize,
        reverse_bits ( ) -> isize,
        to_be ( ) -> isize,
        to_le ( ) -> isize,
        saturating_add ( rhs: isize ) -> isize,
        saturating_sub ( rhs: isize ) -> isize,
        saturating_neg ( ) -> isize,
        saturating_abs ( ) -> isize,
        saturating_mul ( rhs: isize ) -> isize,
        saturating_div ( rhs: isize ) -> isize,
        saturating_pow ( exp: u32 ) -> isize,
        wrapping_add ( rhs: isize ) -> isize,
        wrapping_sub ( rhs: isize ) -> isize,
        wrapping_mul ( rhs: isize ) -> isize,
        wrapping_div ( rhs: isize ) -> isize,
        wrapping_div_euclid ( rhs: isize ) -> isize,
        wrapping_rem ( rhs: isize ) -> isize,
        wrapping_rem_euclid ( rhs: isize ) -> isize,
        wrapping_neg ( ) -> isize,
        wrapping_shl ( rhs: u32 ) -> isize,
        wrapping_shr ( rhs: u32 ) -> isize,
        wrapping_abs ( ) -> isize,
        unsigned_abs ( ) -> usize,
        wrapping_pow ( exp: u32 ) -> isize,
        pow ( exp: u32 ) -> isize,
        div_euclid ( rhs: isize ) -> isize,
        rem_euclid ( rhs: isize ) -> isize,
        abs ( ) -> isize,
        signum ( ) -> isize,
        is_positive ( ) -> bool,
        is_negative ( ) -> bool,
        to_be_bytes ( ) -> [u8; 8],
        to_le_bytes ( ) -> [u8; 8],
        to_ne_bytes ( ) -> [u8; 8],
    }
);

impl_nd_passthrough_methods!(
    u128 : {
        count_ones ( ) -> u32,
        count_zeros ( ) -> u32,
        leading_zeros ( ) -> u32,
        trailing_zeros ( ) -> u32,
        leading_ones ( ) -> u32,
        trailing_ones ( ) -> u32,
        rotate_left ( n: u32 ) -> u128,
        rotate_right ( n: u32 ) -> u128,
        swap_bytes ( ) -> u128,
        reverse_bits ( ) -> u128,
        to_be ( ) -> u128,
        to_le ( ) -> u128,
        saturating_add ( rhs: u128 ) -> u128,
        saturating_sub ( rhs: u128 ) -> u128,
        saturating_mul ( rhs: u128 ) -> u128,
        saturating_div ( rhs: u128 ) -> u128,
        saturating_pow ( exp: u32 ) -> u128,
        wrapping_add ( rhs: u128 ) -> u128,
        wrapping_sub ( rhs: u128 ) -> u128,
        wrapping_mul ( rhs: u128 ) -> u128,
        wrapping_div ( rhs: u128 ) -> u128,
        wrapping_div_euclid ( rhs: u128 ) -> u128,
        wrapping_rem ( rhs: u128 ) -> u128,
        wrapping_rem_euclid ( rhs: u128 ) -> u128,
        wrapping_neg ( ) -> u128,
        wrapping_shl ( rhs: u32 ) -> u128,
        wrapping_shr ( rhs: u32 ) -> u128,
        wrapping_pow ( exp: u32 ) -> u128,
        pow ( exp: u32 ) -> u128,
        div_euclid ( rhs: u128 ) -> u128,
        rem_euclid ( rhs: u128 ) -> u128,
        is_power_of_two ( ) -> bool,
        next_power_of_two ( ) -> u128,
        to_be_bytes ( ) -> [u8; 16],
        to_le_bytes ( ) -> [u8; 16],
        to_ne_bytes ( ) -> [u8; 16],
    }
);

impl_nd_passthrough_methods!(
    u64 : {
        count_ones ( ) -> u32,
        count_zeros ( ) -> u32,
        leading_zeros ( ) -> u32,
        trailing_zeros ( ) -> u32,
        leading_ones ( ) -> u32,
        trailing_ones ( ) -> u32,
        rotate_left ( n: u32 ) -> u64,
        rotate_right ( n: u32 ) -> u64,
        swap_bytes ( ) -> u64,
        reverse_bits ( ) -> u64,
        to_be ( ) -> u64,
        to_le ( ) -> u64,
        saturating_add ( rhs: u64 ) -> u64,
        saturating_sub ( rhs: u64 ) -> u64,
        saturating_mul ( rhs: u64 ) -> u64,
        saturating_div ( rhs: u64 ) -> u64,
        saturating_pow ( exp: u32 ) -> u64,
        wrapping_add ( rhs: u64 ) -> u64,
        wrapping_sub ( rhs: u64 ) -> u64,
        wrapping_mul ( rhs: u64 ) -> u64,
        wrapping_div ( rhs: u64 ) -> u64,
        wrapping_div_euclid ( rhs: u64 ) -> u64,
        wrapping_rem ( rhs: u64 ) -> u64,
        wrapping_rem_euclid ( rhs: u64 ) -> u64,
        wrapping_neg ( ) -> u64,
        wrapping_shl ( rhs: u32 ) -> u64,
        wrapping_shr ( rhs: u32 ) -> u64,
        wrapping_pow ( exp: u32 ) -> u64,
        pow ( exp: u32 ) -> u64,
        div_euclid ( rhs: u64 ) -> u64,
        rem_euclid ( rhs: u64 ) -> u64,
        is_power_of_two ( ) -> bool,
        next_power_of_two ( ) -> u64,
        to_be_bytes ( ) -> [u8; 8],
        to_le_bytes ( ) -> [u8; 8],
        to_ne_bytes ( ) -> [u8; 8],
    }
);

impl_nd_passthrough_methods!(
    u32 : {
        count_ones ( ) -> u32,
        count_zeros ( ) -> u32,
        leading_zeros ( ) -> u32,
        trailing_zeros ( ) -> u32,
        leading_ones ( ) -> u32,
        trailing_ones ( ) -> u32,
        rotate_left ( n: u32 ) -> u32,
        rotate_right ( n: u32 ) -> u32,
        swap_bytes ( ) -> u32,
        reverse_bits ( ) -> u32,
        to_be ( ) -> u32,
        to_le ( ) -> u32,
        saturating_add ( rhs: u32 ) -> u32,
        saturating_sub ( rhs: u32 ) -> u32,
        saturating_mul ( rhs: u32 ) -> u32,
        saturating_div ( rhs: u32 ) -> u32,
        saturating_pow ( exp: u32 ) -> u32,
        wrapping_add ( rhs: u32 ) -> u32,
        wrapping_sub ( rhs: u32 ) -> u32,
        wrapping_mul ( rhs: u32 ) -> u32,
        wrapping_div ( rhs: u32 ) -> u32,
        wrapping_div_euclid ( rhs: u32 ) -> u32,
        wrapping_rem ( rhs: u32 ) -> u32,
        wrapping_rem_euclid ( rhs: u32 ) -> u32,
        wrapping_neg ( ) -> u32,
        wrapping_shl ( rhs: u32 ) -> u32,
        wrapping_shr ( rhs: u32 ) -> u32,
        wrapping_pow ( exp: u32 ) -> u32,
        pow ( exp: u32 ) -> u32,
        div_euclid ( rhs: u32 ) -> u32,
        rem_euclid ( rhs: u32 ) -> u32,
        is_power_of_two ( ) -> bool,
        next_power_of_two ( ) -> u32,
        to_be_bytes ( ) -> [u8; 4],
        to_le_bytes ( ) -> [u8; 4],
        to_ne_bytes ( ) -> [u8; 4],
    }
);

impl_nd_passthrough_methods!(
    u16 : {
        count_ones ( ) -> u32,
        count_zeros ( ) -> u32,
        leading_zeros ( ) -> u32,
        trailing_zeros ( ) -> u32,
        leading_ones ( ) -> u32,
        trailing_ones ( ) -> u32,
        rotate_left ( n: u32 ) -> u16,
        rotate_right ( n: u32 ) -> u16,
        swap_bytes ( ) -> u16,
        reverse_bits ( ) -> u16,
        to_be ( ) -> u16,
        to_le ( ) -> u16,
        saturating_add ( rhs: u16 ) -> u16,
        saturating_sub ( rhs: u16 ) -> u16,
        saturating_mul ( rhs: u16 ) -> u16,
        saturating_div ( rhs: u16 ) -> u16,
        saturating_pow ( exp: u32 ) -> u16,
        wrapping_add ( rhs: u16 ) -> u16,
        wrapping_sub ( rhs: u16 ) -> u16,
        wrapping_mul ( rhs: u16 ) -> u16,
        wrapping_div ( rhs: u16 ) -> u16,
        wrapping_div_euclid ( rhs: u16 ) -> u16,
        wrapping_rem ( rhs: u16 ) -> u16,
        wrapping_rem_euclid ( rhs: u16 ) -> u16,
        wrapping_neg ( ) -> u16,
        wrapping_shl ( rhs: u32 ) -> u16,
        wrapping_shr ( rhs: u32 ) -> u16,
        wrapping_pow ( exp: u32 ) -> u16,
        pow ( exp: u32 ) -> u16,
        div_euclid ( rhs: u16 ) -> u16,
        rem_euclid ( rhs: u16 ) -> u16,
        is_power_of_two ( ) -> bool,
        next_power_of_two ( ) -> u16,
        to_be_bytes ( ) -> [u8; 2],
        to_le_bytes ( ) -> [u8; 2],
        to_ne_bytes ( ) -> [u8; 2],
    }
);

impl_nd_passthrough_methods!(
    u8 : {
        count_ones ( ) -> u32,
        count_zeros ( ) -> u32,
        leading_zeros ( ) -> u32,
        trailing_zeros ( ) -> u32,
        leading_ones ( ) -> u32,
        trailing_ones ( ) -> u32,
        rotate_left ( n: u32 ) -> u8,
        rotate_right ( n: u32 ) -> u8,
        swap_bytes ( ) -> u8,
        reverse_bits ( ) -> u8,
        to_be ( ) -> u8,
        to_le ( ) -> u8,
        saturating_add ( rhs: u8 ) -> u8,
        saturating_sub ( rhs: u8 ) -> u8,
        saturating_mul ( rhs: u8 ) -> u8,
        saturating_div ( rhs: u8 ) -> u8,
        saturating_pow ( exp: u32 ) -> u8,
        wrapping_add ( rhs: u8 ) -> u8,
        wrapping_sub ( rhs: u8 ) -> u8,
        wrapping_mul ( rhs: u8 ) -> u8,
        wrapping_div ( rhs: u8 ) -> u8,
        wrapping_div_euclid ( rhs: u8 ) -> u8,
        wrapping_rem ( rhs: u8 ) -> u8,
        wrapping_rem_euclid ( rhs: u8 ) -> u8,
        wrapping_neg ( ) -> u8,
        wrapping_shl ( rhs: u32 ) -> u8,
        wrapping_shr ( rhs: u32 ) -> u8,
        wrapping_pow ( exp: u32 ) -> u8,
        pow ( exp: u32 ) -> u8,
        div_euclid ( rhs: u8 ) -> u8,
        rem_euclid ( rhs: u8 ) -> u8,
        is_power_of_two ( ) -> bool,
        next_power_of_two ( ) -> u8,
        to_be_bytes ( ) -> [u8; 1],
        to_le_bytes ( ) -> [u8; 1],
        to_ne_bytes ( ) -> [u8; 1],
    }
);

impl_nd_passthrough_methods!(
    usize : {
        count_ones ( ) -> u32,
        count_zeros ( ) -> u32,
        leading_zeros ( ) -> u32,
        trailing_zeros ( ) -> u32,
        leading_ones ( ) -> u32,
        trailing_ones ( ) -> u32,
        rotate_left ( n: u32 ) -> usize,
        rotate_right ( n: u32 ) -> usize,
        swap_bytes ( ) -> usize,
        reverse_bits ( ) -> usize,
        to_be ( ) -> usize,
        to_le ( ) -> usize,
        saturating_add ( rhs: usize ) -> usize,
        saturating_sub ( rhs: usize ) -> usize,
        saturating_mul ( rhs: usize ) -> usize,
        saturating_div ( rhs: usize ) -> usize,
        saturating_pow ( exp: u32 ) -> usize,
        wrapping_add ( rhs: usize ) -> usize,
        wrapping_sub ( rhs: usize ) -> usize,
        wrapping_mul ( rhs: usize ) -> usize,
        wrapping_div ( rhs: usize ) -> usize,
        wrapping_div_euclid ( rhs: usize ) -> usize,
        wrapping_rem ( rhs: usize ) -> usize,
        wrapping_rem_euclid ( rhs: usize ) -> usize,
        wrapping_neg ( ) -> usize,
        wrapping_shl ( rhs: u32 ) -> usize,
        wrapping_shr ( rhs: u32 ) -> usize,
        wrapping_pow ( exp: u32 ) -> usize,
        pow ( exp: u32 ) -> usize,
        div_euclid ( rhs: usize ) -> usize,
        rem_euclid ( rhs: usize ) -> usize,
        is_power_of_two ( ) -> bool,
        next_power_of_two ( ) -> usize,
        to_be_bytes ( ) -> [u8; 8],
        to_le_bytes ( ) -> [u8; 8],
        to_ne_bytes ( ) -> [u8; 8],
    }
);

macro_rules! impl_nd_alg_op(
    ( $trt:ident, $op:tt, $fun:ident ) => {
        impl<'a, T1, S1, T2, S2, T3, S3, D>
            $trt<&'a NDBase<T2, S2, D>> for &'a NDBase<T1, S1, D>
        where
            T1: Copy + ScalarOperand
                + $trt<T2, Output=T3>
                + $trt<&'a nd::ArrayBase<S2, D>, Output=nd::ArrayBase<S3, D>>,
            &'a nd::ArrayBase<S1, D>:
                $trt<T2, Output=nd::ArrayBase<S3, D>>
                + $trt<&'a nd::ArrayBase<S2, D>, Output=nd::ArrayBase<S3, D>>,
            S1: Data<Elem=T1>,
            T2: Copy + ScalarOperand
                + $trt<T1, Output=T3>
                + $trt<&'a nd::ArrayBase<S1, D>, Output=nd::ArrayBase<S3, D>>,
            S2: Data<Elem=T2>,
            S3: Data<Elem=T3>,
            D: Dimension,
        {
            type Output = NDBase<T3, S3, D>;

            fn $fun(self, other: &'a NDBase<T2, S2, D>) -> NDBase<T3, S3, D> {
                return match self {
                    NDBase::S(lhs_s) => match other {
                        NDBase::S(rhs_s) => NDBase::S(*lhs_s $op *rhs_s),
                        NDBase::A(rhs_a) => NDBase::A(*lhs_s $op rhs_a),
                    },
                    NDBase::A(lhs_a) => match other {
                        NDBase::S(rhs_s) => NDBase::A(lhs_a $op *rhs_s),
                        NDBase::A(rhs_a) => NDBase::A(lhs_a $op rhs_a),
                    },
                };
            }
        }

        impl<'a, T1, S1, T2, T3, S3, D> $trt<T2> for &'a NDBase<T1, S1, D>
        where
            T1: Copy + ScalarOperand
                + $trt<T2, Output=T3>,
            &'a nd::ArrayBase<S1, D>: $trt<T2, Output=nd::ArrayBase<S3, D>>,
            S1: Data<Elem=T1>,
            S3: Data<Elem=T3>,
            D: Dimension,
        {
            type Output = NDBase<T3, S3, D>;

            fn $fun(self, rhs: T2) -> NDBase<T3, S3, D> {
                return match self {
                    NDBase::S(lhs_s) => NDBase::S(*lhs_s $op rhs),
                    NDBase::A(lhs_a) => NDBase::A(lhs_a $op rhs),
                };
            }
        }

        // impl<'a, T1, S1, T2, S2, T3, S3, D>
        //     $trt<&'a nd::ArrayBase<S2, D>> for &'a NDBase<T1, S1, D>
        // where
        //     T1: Copy + ScalarOperand
        //         + $trt<&'a nd::ArrayBase<S2, D>, Output=nd::ArrayBase<S3, D>>,
        //     S1: Data<Elem=T1>,
        //     T2: ScalarOperand,
        //     S2: Data<Elem=T2>,
        //     S3: Data<Elem=T3>,
        //     D: Dimension,
        // {
        //     type Output = NDBase<T3, S3, D>;
        //
        //     fn $fun(self, rhs: &'a nd::ArrayBase<S2, D>) -> NDBase<T3, S3, D> {
        //         return match self {
        //             NDBase::S(lhs_s) => NDBase::A(*lhs_s $op rhs),
        //             NDBase::A(lhs_a) => NDBase::A(lhs_a $op rhs),
        //         };
        //     }
        // }
    }
);

impl_nd_alg_op!(Add, +, add);
impl_nd_alg_op!(Sub, -, sub);
impl_nd_alg_op!(Mul, *, mul);
impl_nd_alg_op!(Div, /, div);

pub type ND<T, D> = NDBase<T, OwnedRepr<T>, D>;
pub type ND1<T> = NDBase<T, OwnedRepr<T>, Ix1>;
pub type ND2<T> = NDBase<T, OwnedRepr<T>, Ix2>;
pub type ND3<T> = NDBase<T, OwnedRepr<T>, Ix3>;
pub type ND4<T> = NDBase<T, OwnedRepr<T>, Ix4>;
pub type ND5<T> = NDBase<T, OwnedRepr<T>, Ix5>;
pub type ND6<T> = NDBase<T, OwnedRepr<T>, Ix6>;
pub type NDD<T> = NDBase<T, OwnedRepr<T>, IxDyn>;

pub type NDView<'a, T, D> = NDBase<T, ViewRepr<&'a T>, D>;
pub type NDView1<'a, T> = NDBase<T, ViewRepr<&'a T>, Ix1>;
pub type NDView2<'a, T> = NDBase<T, ViewRepr<&'a T>, Ix2>;
pub type NDView3<'a, T> = NDBase<T, ViewRepr<&'a T>, Ix3>;
pub type NDView4<'a, T> = NDBase<T, ViewRepr<&'a T>, Ix4>;
pub type NDView5<'a, T> = NDBase<T, ViewRepr<&'a T>, Ix5>;
pub type NDView6<'a, T> = NDBase<T, ViewRepr<&'a T>, Ix6>;
pub type NDViewD<'a, T> = NDBase<T, ViewRepr<&'a T>, IxDyn>;

