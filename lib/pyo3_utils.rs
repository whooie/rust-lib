//! Provides traits and functions for managing certain parts of the `pyo3`
//! system succinctly.

use std::{
    error,
    fmt,
};
use pyo3::{
    types as pytypes,
    conversion as pyconv,
};
#[allow(unused_imports)]
use crate::{
    error::ErrMsg,
};

/// Trait to get and extract values from a `PyDict`. Provides two variants of
/// operation. The first returns the object if it exists and an `Err` if it
/// doesn't or if the conversion to a Rust type fails. The other returns an
/// `Option` and only errors if the type conversion will fail.
pub trait GetExtract {
    fn get_extract<'a, T, K, E>(&'a self, key: K, missing_err: E, conv_err: E)
        -> Result<T, E>
        where T: pyconv::FromPyObject<'a>,
              K: pyconv::ToPyObject,
              E: error::Error + fmt::Display;

    fn get_extract_opt<'a, T, K, E>(&'a self, key: K, conv_err: E)
        -> Result<Option<T>, E>
        where T: pyconv::FromPyObject<'a>,
              K: pyconv::ToPyObject,
              E: error::Error + fmt::Display;
}

impl GetExtract for pytypes::PyDict {
    fn get_extract<'a, T, K, E>(&'a self, key: K, missing_err: E, conv_err: E)
        -> Result<T, E>
        where T: pyconv::FromPyObject<'a>,
              K: pyconv::ToPyObject,
              E: error::Error + fmt::Display,
    {
        let res: Result<T, E> = match self.get_item(key) {
            Some(Obj) => Ok(Obj.extract().map_err(|_err| conv_err)?),
            None => Err(missing_err),
        };
        return res;
    }

    fn get_extract_opt<'a, T, K, E>(&'a self, key: K, conv_err: E)
        -> Result<Option<T>, E>
        where T: pyconv::FromPyObject<'a>,
              K: pyconv::ToPyObject,
              E: error::Error + fmt::Display,
    {
        let res: Option<T> = match self.get_item(key) {
            Some(Obj) => Some(Obj.extract().map_err(|_err| conv_err)?),
            None => None,
        };
        return Ok(res);
    }
}

/// Implements `From<impl ErrMsg>` (see [ErrMsg]) for `PyErr`.
#[macro_export]
macro_rules! mkintopyerr {
    ( $name:ident : [$( $var:ident => $msg:literal ),+] ) => {
        impl From<$name> for pyo3::prelude::PyErr {
            fn from(err: $name) -> Self {
                return pyo3::exceptions::PyException::new_err(err.msg());
            }
        }
    }
}

