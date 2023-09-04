#![allow(non_snake_case)]
#![allow(dead_code)]
#![allow(non_upper_case_globals)]
#![allow(clippy::needless_return)]

//! This crate is a collection of useful types, traits, and functions mostly
//! meant for light, general-purpose scientific use with `ndarray`, `plotly`,
//! `pyo3`, and some others.

pub mod config;
pub mod error;
pub mod utils;
pub mod phys;

#[cfg(feature = "nd")]
pub mod nd;

#[cfg(feature = "ndarray-utils")]
pub mod nd_utils;

#[cfg(feature = "plotting")]
pub mod plotdefs;

#[cfg(feature = "plotting-plotly")]
pub mod plotlydefs;

#[cfg(feature = "pyo3-utils")]
pub mod pyo3_utils;

#[cfg(feature = "math")]
pub mod math;

#[cfg(feature = "zx")]
pub mod zx;

