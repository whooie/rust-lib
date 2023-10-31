//! Provides definitions for loading and verifying values from common config
//! file formats.
//!
//! Structures aim to preserve the default type structures of their respective
//! config [deserializer][serde] crates.
//!
//! The important traits are made public for custom applications and newtype
//! wrapping, but the following formats are directly supported:
//! - [TOML][toml]
//! - [JSON][serde_json]
//! - [YAML][serde_yaml][^1]
//!
//! [^1]: No tagged values and only string keys

use std::{
    collections::{ HashMap, HashSet },
    fmt,
    fs,
    io::Write,
    iter::Peekable,
    path::Path,
    str::FromStr,
};
use toml;
use serde_json as json;
use serde_yaml as yaml;
use serde::{ Deserialize, Serialize };
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ConfigError {
    #[error("config: invalid value for key '{0}': expected value to satisfy '{1}' but got {2}")]
    InvalidValue(String, String, String),

    #[error("config: encountered incompatible structure")]
    IncompatibleStructure,

    #[error("config: missing key '{0}'")]
    MissingKey(String),

    #[error("config: failed to convert type of value at key '{0}'")]
    FailedTypeConversion(String),

    #[error("config: couldn't read file '{0}'")]
    FileRead(String),

    #[error("config: couldn't parse file '{0}'")]
    FileParse(String),

    #[error("config: couldn't parse string")]
    StrParse,

    #[error("config: couldn't open file '{0}': {1}")]
    FileOpen(String, String),

    #[error("config: couldn't write to file '{0}': {1}")]
    FileWrite(String, String),

    #[error("config: TOML error '{0}'")]
    TomlError(#[from] toml::ser::Error),

    #[error("config: JSON error '{0}'")]
    JsonError(#[from] json::Error),

    #[error("config: YAML error '{0}'")]
    YamlError(#[from] yaml::Error),
}
pub type ConfigResult<T> = Result<T, ConfigError>;

/// Represents a constraint on the type and/or value of an item in a config.
pub trait ValueVerifier<V> {
    /// Return `true` if `value` matches the type specification, `false`
    /// otherwise.
    fn verify(&self, value: &V) -> bool;

    /// Return `value` if it matches the type specification, `err` otherwise.
    fn verify_ok_or<E>(&self, value: V, err: E) -> Result<V, E> {
        self.verify(&value).then_some(value).ok_or(err)
    }
}

/// Config value type specification.
#[derive(Clone, Debug)]
pub enum Verifier {
    /// Null value
    Null,

    /// Boolean value
    Bool,

    /// Integer value (default `i64`)
    Int,

    /// A range of integer (default `i64`) values
    IntRange {
        min: i64,
        max: i64,
        incl_min: bool,
        incl_max: bool,
    },

    /// A discrete collection of integer (default `i64`) values
    IntColl(HashSet<i64>),

    /// Floating-point value (default `f64`)
    Float,

    /// A range of floating-point (default `f64`) values
    FloatRange {
        min: f64,
        max: f64,
        incl_min: bool,
        incl_max: bool,
    },

    /// An integer or floating-point value
    Number,

    /// A date-time value
    Datetime,

    /// An array of any length or type
    Array,

    /// An array with constrained values.
    ///
    /// Using `finite = true` will impose constraints on the length and position
    /// of each value in the array; `finite = false` will permit any collection
    /// of values that satisfies at least one of the contained constraints.
    ArrayForm {
        forms: Vec<Verifier>,
        finite: bool,
    },

    /// String value
    Str,

    /// A discrete collection of string values
    StrColl(HashSet<String>),

    /// A sub-table
    Table,

    /// A sub-table with constrained values.
    TableForm(HashMap<String, Verifier>),

    /// Any non-table value
    Any,

    /// Any value, including a sub-table
    AnyTable,
}

fn indent_block(instr: &str, n: usize) -> String {
    instr.split('\n')
        .map(|line| "  ".repeat(n) + line)
        .collect::<Vec<String>>()
        .join("\n")
}

impl fmt::Display for Verifier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Null => write!(f, "Null"),
            Self::Bool => write!(f, "Bool"),
            Self::Int => write!(f, "Int"),
            Self::IntRange { min, max, incl_min, incl_max } => {
                write!(f, "IntRange{}", if *incl_min { "[" } else { "(" })?;
                min.fmt(f)?;
                write!(f, ", ")?;
                max.fmt(f)?;
                write!(f, "{}", if *incl_max { "]" } else { ")" })?;
                Ok(())
            },
            Self::IntColl(coll) => {
                let items: String
                    = coll.iter()
                    .map(|i| i.to_string())
                    .collect::<Vec<String>>()
                    .join(", ");
                write!(f, "IntCollection{{{}}}", items)?;
                Ok(())
            },
            Self::Float => write!(f, "Float"),
            Self::FloatRange { min, max, incl_min, incl_max } => {
                write!(f, "FloatRange{}", if *incl_min { "[" } else { "(" })?;
                min.fmt(f)?;
                write!(f, ", ")?;
                max.fmt(f)?;
                write!(f, "{}", if *incl_max { "]" } else { ")" })?;
                Ok(())
            },
            Self::Number => write!(f, "Number"),
            Self::Datetime => write!(f, "Datetime"),
            Self::Array => write!(f, "Array"),
            Self::ArrayForm { forms, finite } => {
                let items: String
                    = forms.iter()
                    .map(|f| f.to_string())
                    .collect::<Vec<String>>()
                    .join(", ");
                write!(f, "ArrayForm([{}], finite={})", items, finite)?;
                Ok(())
            },
            Self::Str => write!(f, "Str"),
            Self::StrColl(coll) => {
                let items: String
                    = coll.iter()
                    .map(|s| s.as_ref())
                    .collect::<Vec<&str>>()
                    .join(", ");
                write!(f, "StrCollection{{{}}}", items)?;
                Ok(())
            },
            Self::Table => write!(f, "Table"),
            Self::TableForm(tab) => {
                let items: String
                    = indent_block(
                        &tab.iter()
                            .map(|(s, v)| format!("{} : {}", s, v))
                            .collect::<Vec<String>>()
                            .join(",\n"),
                        1,
                    );
                write!(f, "TableForm {{\n{}\n}}", items)?;
                Ok(())
            },
            Self::Any => write!(f, "Any"),
            Self::AnyTable => write!(f, "AnyTable"),
        }
    }
}

impl ValueVerifier<toml::Value> for Verifier {
    fn verify(&self, value: &toml::Value) -> bool {
        use toml::Value;

        match (self, value) {
            (Self::Bool, Value::Boolean(_)) => true,
            (Self::Int, Value::Integer(_)) => true,
            (
                Self::IntRange { min, max, incl_min, incl_max },
                Value::Integer(i),
            ) => {
                let in_min: bool
                    = if *incl_min { *i >= *min } else { *i > *min };
                let in_max: bool
                    = if *incl_max { *i <= *max } else { *i < *max };
                in_min && in_max
            },
            (Self::IntColl(coll), Value::Integer(i)) => coll.contains(i),
            (Self::Float, Value::Float(_)) => true,
            (
                Self::FloatRange { min, max, incl_min, incl_max },
                Value::Float(f),
            ) => {
                let in_min: bool
                    = if *incl_min { *f >= *min } else { *f > *min };
                let in_max: bool
                    = if *incl_max { *f <= *max } else { *f < *max };
                in_min && in_max
            },
            (Self::Number, Value::Integer(_))
                | (Self::Number, Value::Float(_))
                => true,
            (Self::Datetime, Value::Datetime(_)) => true,
            (Self::Array, Value::Array(_)) => true,
            (Self::ArrayForm { forms, finite }, Value::Array(a)) => {
                if *finite {
                    forms.len() == a.len()
                        && forms.iter().zip(a)
                            .all(|(fk, ak)| fk.verify(ak))
                } else {
                    a.iter()
                        .all(|ak| forms.iter().any(|fk| fk.verify(ak)))
                }
            },
            (Self::Str, Value::String(_)) => true,
            (Self::StrColl(coll), Value::String(s)) => coll.contains(s),
            (Self::Table, Value::Table(_)) => true,
            (Self::TableForm(forms), Value::Table(tab)) => {
                forms.iter()
                    .all(|(s, ver)| {
                        tab.get(s).is_some_and(|val| ver.verify(val))
                    })
            },
            (Self::Any, Value::Boolean(_))
                | (Self::Any, Value::Float(_))
                | (Self::Any, Value::Datetime(_))
                | (Self::Any, Value::Array(_))
                => true,
            (Self::AnyTable, _) => true,
            _ => false,
        }
    }
}

impl ValueVerifier<json::Value> for Verifier {
    fn verify(&self, value: &json::Value) -> bool {
        use serde_json::Value;

        match (self, value) {
            (Self::Null, Value::Null) => true,
            (Self::Bool, Value::Bool(_)) => true,
            (Self::Int, Value::Number(n)) => n.is_i64() || n.is_u64(),
            (
                Self::IntRange { min, max, incl_min, incl_max },
                Value::Number(n),
            ) => {
                if let Some(i) = n.as_i64() {
                    let in_min: bool
                        = if *incl_min { i >= *min } else { i > *min };
                    let in_max: bool
                        = if *incl_max { i <= *max } else { i < *max };
                    in_min && in_max
                } else if let Some(u) = n.as_u64() {
                    let u: i128 = u.into();
                    let min: i128 = (*min).into();
                    let max: i128 = (*max).into();
                    let in_min: bool
                        = if *incl_min { u >= min } else { u > min };
                    let in_max: bool
                        = if *incl_max { u <= max } else { u < max };
                    in_min && in_max
                } else {
                    false
                }
            },
            (Self::Float, Value::Number(n)) => n.is_f64(),
            (
                Self::FloatRange { min, max, incl_min, incl_max },
                Value::Number(n),
            ) => {
                if let Some(f) = n.as_f64() {
                    let in_min: bool
                        = if *incl_min { f >= *min } else { f > *min };
                    let in_max: bool
                        = if *incl_max { f <= *max } else { f < *max };
                    in_min && in_max
                } else {
                    false
                }
            },
            (Self::Array, Value::Array(_)) => true,
            (Self::ArrayForm { forms, finite }, Value::Array(a)) => {
                if *finite {
                    forms.len() == a.len()
                        && forms.iter().zip(a)
                            .all(|(fk, ak)| fk.verify(ak))
                } else {
                    a.iter()
                        .all(|ak| forms.iter().any(|fk| fk.verify(ak)))
                }
            },
            (Self::Str, Value::String(_)) => true,
            (Self::StrColl(coll), Value::String(s)) => coll.contains(s),
            (Self::Table, Value::Object(_)) => true,
            (Self::TableForm(forms), Value::Object(obj)) => {
                forms.iter()
                    .all(|(s, ver)| {
                        obj.get(s).is_some_and(|val| ver.verify(val))
                    })
            },
            (Self::Any, Value::Bool(_))
                | (Self::Any, Value::Number(_))
                | (Self::Any, Value::Array(_))
                | (Self::Any, Value::String(_))
                => true,
            (Self::AnyTable, _) => true,
            _ => false,
        }
    }
}

impl ValueVerifier<yaml::Value> for Verifier {
    fn verify(&self, value: &yaml::Value) -> bool {
        use serde_yaml::Value;

        match (self, value) {
            (Self::Null, Value::Null) => true,
            (Self::Bool, Value::Bool(_)) => true,
            (Self::Int, Value::Number(n)) => n.is_i64() || n.is_u64(),
            (
                Self::IntRange { min, max, incl_min, incl_max },
                Value::Number(n),
            ) => {
                if let Some(i) = n.as_i64() {
                    let in_min: bool
                        = if *incl_min { i >= *min } else { i > *min };
                    let in_max: bool
                        = if *incl_max { i <= *max } else { i < *max };
                    in_min && in_max
                } else if let Some(u) = n.as_u64() {
                    let u: i128 = u.into();
                    let min: i128 = (*min).into();
                    let max: i128 = (*max).into();
                    let in_min: bool
                        = if *incl_min { u >= min } else { u > min };
                    let in_max: bool
                        = if *incl_max { u <= max } else { u < max };
                    in_min && in_max
                } else {
                    false
                }
            },
            (Self::Float, Value::Number(n)) => n.is_f64(),
            (
                Self::FloatRange { min, max, incl_min, incl_max },
                Value::Number(n),
            ) => {
                if let Some(f) = n.as_f64() {
                    let in_min: bool
                        = if *incl_min { f >= *min } else { f > *min };
                    let in_max: bool
                        = if *incl_max { f <= *max } else { f < *max };
                    in_min && in_max
                } else {
                    false
                }
            },
            (Self::Array, Value::Sequence(_)) => true,
            (Self::ArrayForm { forms, finite }, Value::Sequence(a)) => {
                if *finite {
                    forms.len() == a.len()
                        && forms.iter().zip(a)
                            .all(|(fk, ak)| fk.verify(ak))
                } else {
                    a.iter()
                        .all(|ak| forms.iter().any(|fk| fk.verify(ak)))
                }
            },
            (Self::Str, Value::String(_)) => true,
            (Self::StrColl(coll), Value::String(s)) => coll.contains(s),
            (Self::Table, Value::Mapping(_)) => true,
            (Self::TableForm(forms), Value::Mapping(map)) => {
                forms.iter()
                    .all(|(s, ver)| {
                        map.get(s).is_some_and(|val| ver.verify(val))
                    })
            },
            (Self::Any, Value::Bool(_))
                | (Self::Any, Value::Number(_))
                | (Self::Any, Value::Sequence(_))
                | (Self::Any, Value::String(_))
                => true,
            (Self::AnyTable, Value::Bool(_))
                | (Self::AnyTable, Value::Number(_))
                | (Self::AnyTable, Value::Sequence(_))
                | (Self::AnyTable, Value::String(_))
                | (Self::AnyTable, Value::Mapping(_))
                => true,
            _ => false,
        }
    }
}

/// An item in a config specification, either a type/value specification for
/// single values or a set of specifications for a sub-table.
#[derive(Clone, Debug)]
pub enum ConfigSpecItem<U, V>
where U: ValueVerifier<V>
{
    Value(U),
    Table(ConfigSpec<U, V>),
}

impl<U, V> From<U> for ConfigSpecItem<U, V>
where U: ValueVerifier<V>
{
    fn from(ver: U) -> Self { Self::Value(ver) }
}

impl<U, V> From<ConfigSpec<U, V>> for ConfigSpecItem<U, V>
where U: ValueVerifier<V>
{
    fn from(confspec: ConfigSpec<U, V>) -> Self { Self::Table(confspec) }
}

/// Sugared [`HashMap`] representing a specification for the values and
/// structure of a formatted config file.
#[derive(Clone, Debug)]
pub struct ConfigSpec<U, V>
where U: ValueVerifier<V>
{
    spec: HashMap<String, ConfigSpecItem<U, V>>,
    config_values: std::marker::PhantomData<V>,
}

impl<U, V> FromIterator<(String, ConfigSpecItem<U, V>)> for ConfigSpec<U, V>
where U: ValueVerifier<V>
{
    fn from_iter<I>(iter: I) -> Self
    where I: IntoIterator<Item = (String, ConfigSpecItem<U, V>)>
    {
        Self {
            spec: iter.into_iter().collect(),
            config_values: std::marker::PhantomData,
        }
    }
}

impl<U, V> Default for ConfigSpec<U, V>
where U: ValueVerifier<V>
{
    fn default() -> Self {
        Self { spec: HashMap::new(), config_values: std::marker::PhantomData }
    }
}

impl<U, V> ConfigSpec<U, V>
where U: ValueVerifier<V>
{
    /// Create a new, empty config specification.
    pub fn new() -> Self { Self::default() }
}

impl<U, V> AsRef<HashMap<String, ConfigSpecItem<U, V>>> for ConfigSpec<U, V>
where U: ValueVerifier<V>
{
    fn as_ref(&self) -> &HashMap<String, ConfigSpecItem<U, V>> { &self.spec }
}

impl<U, V> AsMut<HashMap<String, ConfigSpecItem<U, V>>> for ConfigSpec<U, V>
where U: ValueVerifier<V>
{
    fn as_mut(&mut self) -> &mut HashMap<String, ConfigSpecItem<U, V>> {
        &mut self.spec
    }
}

/// Represents a set of constraints on the values and structure of a config.
pub trait ConfigVerifier<V> {
    /// Error type.
    type Error;

    /// Output verified record type.
    type Record;

    /// Verify a config-level value, consuming and re-returning if it passes
    /// inspection.
    fn verify(&self, value: V) -> Result<V, Self::Error>;

    /// Verify a value and convert to a permanent type.
    fn verify_into(&self, value: V) -> Result<Self::Record, Self::Error>;
}

impl ConfigVerifier<toml::Value> for ConfigSpec<Verifier, toml::Value> {
    type Error = ConfigError;
    type Record = Config<toml::Table, toml::Value>;

    fn verify(&self, value: toml::Value) -> ConfigResult<toml::Value> {
        use toml::Value;

        if let Value::Table(mut tab) = value {
            let mut data = toml::Table::new();
            let mut val: Value;
            let mut valstr: String;
            for (key, spec_item) in self.spec.iter() {
                val = match (spec_item, tab.remove(key)) {
                    (ConfigSpecItem::Table(subspec), Some(v)) => {
                        subspec.verify(v)
                    },
                    (ConfigSpecItem::Value(ver), Some(v)) => {
                        valstr = v.to_string();
                        ver.verify_ok_or(
                            v,
                            ConfigError::InvalidValue(
                                key.clone(),
                                ver.to_string(),
                                valstr,
                            )
                        )
                    },
                    _ => Err(ConfigError::MissingKey(key.clone())),
                }?;
                data.insert(key.clone(), val);
            }
            Ok(Value::Table(data))
        } else {
            Err(ConfigError::IncompatibleStructure)
        }
    }

    fn verify_into(&self, value: toml::Value) -> ConfigResult<Self::Record> {
        self.verify(value)
            .map(|v| Config { data: v.try_into().unwrap() })
    }
}

pub type JsonObject = json::Map<String, json::Value>;

impl ConfigVerifier<json::Value> for ConfigSpec<Verifier, json::Value> {
    type Error = ConfigError;
    type Record = Config<JsonObject, json::Value>;

    fn verify(&self, value: json::Value) -> ConfigResult<json::Value> {
        use json::Value;

        if let Value::Object(mut obj) = value {
            let mut data: json::Map<String, Value> = json::Map::new();
            let mut val: Value;
            let mut valstr: String;
            for (key, spec_item) in self.spec.iter() {
                val = match (spec_item, obj.remove(key)) {
                    (ConfigSpecItem::Table(subspec), Some(v)) => {
                        subspec.verify(v)
                    },
                    (ConfigSpecItem::Value(ver), Some(v)) => {
                        valstr = v.to_string();
                        ver.verify_ok_or(
                            v,
                            ConfigError::InvalidValue(
                                key.clone(),
                                ver.to_string(),
                                valstr,
                            )
                        )
                    },
                    _ => Err(ConfigError::MissingKey(key.clone())),
                }?;
                data.insert(key.clone(), val);
            }
            Ok(Value::Object(data))
        } else {
            Err(ConfigError::IncompatibleStructure)
        }
    }

    fn verify_into(&self, value: json::Value) -> ConfigResult<Self::Record> {
        self.verify(value)
            .map(|v| {
                let json::Value::Object(obj) = v else { unreachable!() };
                Config { data: obj }
            })
    }
}

impl ConfigVerifier<yaml::Value> for ConfigSpec<Verifier, yaml::Value> {
    type Error = ConfigError;
    type Record = Config<yaml::Mapping, yaml::Value>;

    fn verify(&self, value: yaml::Value) -> ConfigResult<yaml::Value> {
        use yaml::Value;

        if let Value::Mapping(mut map) = value {
            let mut data = yaml::Mapping::new();
            let mut val: Value;
            let mut valstr: String;
            for (key, spec_item) in self.spec.iter() {
                val = match (spec_item, map.remove(key)) {
                    (ConfigSpecItem::Table(subspec), Some(v)) => {
                        subspec.verify(v)
                    },
                    (ConfigSpecItem::Value(ver), Some(v)) => {
                        valstr = format!("{:?}", v);
                        ver.verify_ok_or(
                            v,
                            ConfigError::InvalidValue(
                                key.clone(),
                                ver.to_string(),
                                valstr,
                            )
                        )
                    },
                    _ => Err(ConfigError::MissingKey(key.clone())),
                }?;
                data.insert(key.clone().into(), val);
            }
            Ok(Value::Mapping(data))
        } else {
            Err(ConfigError::IncompatibleStructure)
        }
    }

    fn verify_into(&self, value: yaml::Value) -> ConfigResult<Self::Record> {
        self.verify(value)
            .map(|v| {
                let yaml::Value::Mapping(map) = v else { unreachable!() };
                Config { data: map }
            })
    }
}

/// Shorthand for creating [`ConfigSpec`]s and [`ConfigSpecItem`]s.
#[macro_export]
macro_rules! confspec {
    ( Null ) => {
        $crate::config::Verifier::Null.into()
    };
    ( Bool ) => {
        $crate::config::Verifier::Bool.into()
    };
    ( Int ) => {
        $crate::config::Verifier::Int.into()
    };
    ( IntRange { ( $min:expr, $max:expr ) } ) => {
        $crate::config::Verifier::IntRange {
            min: $min,
            max: $max,
            incl_min: false,
            incl_max: false,
        }
        .into()
    };
    ( IntRange { =( $min:expr, $max:expr ) } ) => {
        $crate::config::Verifier::IntRange {
            min: $min,
            max: $max,
            incl_min: true,
            incl_max: false,
        }
        .into()
    };
    ( IntRange { ( $min:expr, $max:expr )= } ) => {
        $crate::config::Verifier::IntRange {
            min: $min,
            max: $max,
            incl_min: false,
            incl_max: true,
        }
        .into()
    };
    ( IntRange { =( $min:expr, $max:expr )= } ) => {
        $crate::config::Verifier::IntRange {
            min: $min,
            max: $max,
            incl_min: true,
            incl_max: true,
        }
        .into()
    };
    ( IntColl { $( $i:expr ),* $(,)? } ) => {
        $crate::config::Verifier::IntColl(
            std::collections::HashSet::from_iter([$( $i ),*])
        )
        .into()
    };
    ( Float ) => {
        $crate::config::Verifier::Float.into()
    };
    ( FloatRange { ( $min:expr, $max:expr ) } ) => {
        $crate::config::Verifier::FloatRange {
            min: $min,
            max: $max,
            incl_min: false,
            incl_max: false,
        }
        .into()
    };
    ( FloatRange { =( $min:expr, $max:expr ) } ) => {
        $crate::config::Verifier::FloatRange {
            min: $min,
            max: $max,
            incl_min: true,
            incl_max: false,
        }
        .into()
    };
    ( FloatRange { ( $min:expr, $max:expr )= } ) => {
        $crate::config::Verifier::FloatRange {
            min: $min,
            max: $max,
            incl_min: false,
            incl_max: true,
        }
        .into()
    };
    ( FloatRange { =( $min:expr, $max:expr )= } ) => {
        $crate::config::Verifier::FloatRange {
            min: $min,
            max: $max,
            incl_min: true,
            incl_max: true,
        }
        .into()
    };
    ( Number ) => {
        $crate::config::Verifier::Number.into()
    };
    ( Datetime ) => {
        $crate::config::Verifier::Datetime.into()
    };
    ( Array ) => {
        $crate::config::Verifier::Array.into()
    };
    ( ArrayForm { [ $( $form:expr ),* $(,)? ], finite: $f:expr $(,)? } ) => {
        $crate::config::Verifier::ArrayForm {
            forms: vec![ $( $form ),* ],
            finite: $f,
        }
        .into()
    };
    ( Str ) => {
        $crate::config::Verifier::Str.into()
    };
    ( StrColl { $( $s:expr ),* $(,)? } ) => {
        $crate::config::Verifier::StrColl(
            std::collections::HashSet::from_iter([$( $s.to_string() ),*])
        )
        .into()
    };
    ( Table { $( $key:expr => $spec:expr ),* $(,)? } ) => {
        $crate::config::ConfigSpec::from_iter([
            $( ($key.to_string(), $spec) ),*
        ])
        .into()
    };
    ( TableForm { $( $key:expr => $spec:expr ),* $(,)? } ) => {
        $crate::config::Verifier::TableForm(
            HashMap::from_iter([
                $( ($key.to_string(), $spec) ),*
            ])
        )
        .into()
    };
    ( Any ) => {
        $crate::config::Verifier::Any.into()
    };
    ( AnyTable ) => {
        $crate::config::Verifier::AnyTable.into()
    };
    ( { $( $key:expr => $spec:expr ),* $(,)? } ) => {
        $crate::config::ConfigSpec::from_iter([
            $( ($key.to_string(), $spec) ),*
        ])
    };
}

/// Convenience accessor/modifier methods on a nested map-like structure where
/// string keys are mapped to `Self::Value`.
///
/// See [`toml::Table`]/[`toml::Value`] and [`json::Map`]/[`json::Value`] for
/// examples.
///
/// Values accessed through these methods are done so using "key paths", which
/// are either iterables of `&str` keys, each accessing a single nested level in
/// the structure, or a single `&str` containing the items of the path
/// deliminated by `Self::DELIM`.
pub trait InternalRecord {
    /// Record value type.
    type Value;

    /// Assumed key path deliminator.
    const DELIM: char;

    /// Get a reference to the value at the end of a key path, if it exists.
    fn get_path<'a, K>(&self, keys: K) -> Option<&Self::Value>
    where K: IntoIterator<Item = &'a str>;

    /// Get a mutable reference to the value at the end of a key path, if it
    /// exists.
    fn get_mut_path<'a, K>(&mut self, keys: K) -> Option<&mut Self::Value>
    where K: IntoIterator<Item = &'a str>;

    /// Get a reference to the value at the end of a key path, if it exists.
    fn get(&self, keypath: &str) -> Option<&Self::Value> {
        self.get_path(keypath.split(Self::DELIM))
    }

    /// Get a mutable reference to the value at the end of a key path, if it
    /// exists.
    fn get_mut(&mut self, keypath: &str) -> Option<&mut Self::Value> {
        self.get_mut_path(keypath.split(Self::DELIM))
    }

    /// Insert a value at the end of a key path, returning the previous value at
    /// that location if it existed.
    fn insert_path<'a, K, T>(&mut self, keys: K, value: T)
        -> Option<Self::Value>
    where
        K: IntoIterator<Item = &'a str>,
        T: Into<Self::Value>;

    /// Insert a value at the end of a key path, returning the previous value at
    /// that location if it existed.
    fn insert<T>(&mut self, keypath: &str, value: T) -> Option<Self::Value>
    where T: Into<Self::Value>
    {
        self.insert_path(keypath.split(Self::DELIM), value)
    }

    /// Remove a value at the end of a key path if it existed.
    fn remove_path<'a, K>(&mut self, keys: K) -> Option<Self::Value>
    where K: IntoIterator<Item = &'a str>;

    /// Remove a value at the end of a key path if it existed.
    fn remove(&mut self, keypath: &str) -> Option<Self::Value> {
        self.remove_path(keypath.split(Self::DELIM))
    }
}

impl InternalRecord for toml::Table {
    type Value = toml::Value;
    const DELIM: char = '.';

    fn get_path<'a, K>(&self, keys: K) -> Option<&Self::Value>
    where K: IntoIterator<Item = &'a str>
    {
        fn table_get_path<'a, K>(
            table: &toml::Table,
            mut keys: Peekable<K>,
        ) -> Option<&toml::Value>
        where K: Iterator<Item = &'a str>
        {
            if let Some(key) = keys.next() {
                match (table.get(key), keys.peek()) {
                    (Some(toml::Value::Table(subtab)), Some(_)) => {
                        table_get_path(subtab, keys)
                    },
                    (x, None) => x,
                    (Some(_), Some(_)) => None,
                    (None, _) => None,
                }
            } else {
                None
            }
        }
        table_get_path(self, keys.into_iter().peekable())
    }

    fn get_mut_path<'a, K>(&mut self, keys: K) -> Option<&mut Self::Value>
    where K: IntoIterator<Item = &'a str>
    {
        fn table_get_mut_path<'a, K>(
            table: &mut toml::Table,
            mut keys: Peekable<K>,
        ) -> Option<&mut toml::Value>
        where K: Iterator<Item = &'a str>
        {
            if let Some(key) = keys.next() {
                match (table.get_mut(key), keys.peek()) {
                    (Some(toml::Value::Table(subtab)), Some(_)) => {
                        table_get_mut_path(subtab, keys)
                    },
                    (x, None) => x,
                    (Some(_), Some(_)) => None,
                    (None, _) => None,
                }
            } else {
                None
            }
        }
        table_get_mut_path(self, keys.into_iter().peekable())
    }

    /// This method panics if the user tries to `insert` a value at the end of a
    /// path containing a non-table value anywhere before its end.
    fn insert_path<'a, K, T>(&mut self, keys: K, value: T)
        -> Option<Self::Value>
    where
        K: IntoIterator<Item = &'a str>,
        T: Into<Self::Value>,
    {
        fn table_insert_path<'a, K, T>(
            table: &mut toml::Table,
            mut keys: Peekable<K>,
            value: T,
        ) -> Option<toml::Value>
        where
            K: Iterator<Item = &'a str>,
            T: Into<toml::Value>,
        {
            if let Some(key) = keys.next() {
                match (table.get_mut(key), keys.peek()) {
                    (Some(toml::Value::Table(subtab)), Some(_)) => {
                        table_insert_path(subtab, keys, value)
                    },
                    (None, Some(_)) => {
                        table.insert(
                            key.to_string(), toml::Table::new().into());
                        table_insert_path(
                            table.get_mut(key).unwrap()
                                .as_table_mut().unwrap(),
                            keys,
                            value,
                        )
                    },
                    (Some(x), None) => {
                        Some(std::mem::replace(x, value.into()))
                    },
                    (None, None) => {
                        table.insert(key.to_string(), value.into())
                    },
                    (Some(_), Some(_)) => {
                        panic!(
                            "InternalRecord for toml::Table: encountered \
                            non-empty values in the middle of a path insertion"
                        );
                    },
                }
            } else {
                None
            }
        }
        table_insert_path(self, keys.into_iter().peekable(), value)
    }

    fn remove_path<'a, K>(&mut self, keys: K) -> Option<Self::Value>
    where K: IntoIterator<Item = &'a str>
    {
        fn table_remove_path<'a, K>(
            table: &mut toml::Table,
            mut keys: Peekable<K>,
        ) -> Option<toml::Value>
        where K: Iterator<Item = &'a str>
        {
            if let Some(key) = keys.next() {
                match (table.get_mut(key), keys.peek()) {
                    (Some(toml::Value::Table(subtab)), Some(_)) => {
                        table_remove_path(subtab, keys)
                    },
                    (_, None) => table.remove(key),
                    (Some(_), Some(_)) => None,
                    (None, _) => None,
                }
            } else {
                None
            }
        }
        table_remove_path(self, keys.into_iter().peekable())
    }
}

impl InternalRecord for json::Map<String, json::Value> {
    type Value = json::Value;
    const DELIM: char = '.';

    fn get_path<'a, K>(&self, keys: K) -> Option<&Self::Value>
    where K: IntoIterator<Item = &'a str>
    {
        fn object_get_path<'a, K>(
            object: &JsonObject,
            mut keys: Peekable<K>,
        ) -> Option<&json::Value>
        where K: Iterator<Item = &'a str>
        {
            if let Some(key) = keys.next() {
                match (object.get(key), keys.peek()) {
                    (Some(json::Value::Object(subobj)), Some(_)) => {
                        object_get_path(subobj, keys)
                    },
                    (x, None) => x,
                    (Some(_), Some(_)) => None,
                    (None, _) => None,
                }
            } else {
                None
            }
        }
        object_get_path(self, keys.into_iter().peekable())
    }

    fn get_mut_path<'a, K>(&mut self, keys: K) -> Option<&mut Self::Value>
    where K: IntoIterator<Item = &'a str>
    {
        fn object_get_mut_path<'a, K>(
            object: &mut JsonObject,
            mut keys: Peekable<K>,
        ) -> Option<&mut json::Value>
        where K: Iterator<Item = &'a str>
        {
            if let Some(key) = keys.next() {
                match (object.get_mut(key), keys.peek()) {
                    (Some(json::Value::Object(subobj)), Some(_)) => {
                        object_get_mut_path(subobj, keys)
                    },
                    (x, None) => x,
                    (Some(_), Some(_)) => None,
                    (None, _) => None,
                }
            } else {
                None
            }
        }
        object_get_mut_path(self, keys.into_iter().peekable())
    }

    /// This method panics if the user tries to `insert` a value at the end of a
    /// path containing a non-table value anywhere before its end.
    fn insert_path<'a, K, T>(&mut self, keys: K, value: T)
        -> Option<Self::Value>
    where
        K: IntoIterator<Item = &'a str>,
        T: Into<Self::Value>,
    {
        fn object_insert_path<'a, K, T>(
            object: &mut JsonObject,
            mut keys: Peekable<K>,
            value: T,
        ) -> Option<json::Value>
        where
            K: Iterator<Item = &'a str>,
            T: Into<json::Value>,
        {
            if let Some(key) = keys.next() {
                match (object.get_mut(key), keys.peek()) {
                    (Some(json::Value::Object(subobj)), Some(_)) => {
                        object_insert_path(subobj, keys, value)
                    },
                    (None, Some(_)) => {
                        object.insert(
                            key.to_string(), JsonObject::new().into());
                        object_insert_path(
                            object.get_mut(key).unwrap()
                                .as_object_mut().unwrap(),
                            keys,
                            value,
                        )
                    },
                    (Some(x), None) => {
                        Some(std::mem::replace(x, value.into()))
                    },
                    (None, None) => {
                        object.insert(key.to_string(), value.into())
                    },
                    (Some(_), Some(_)) => {
                        panic!(
                            "InternalRecord for json::Map: encountered \
                            non-empty values in the middle of a path insertion"
                        );
                    },
                }
            } else {
                None
            }
        }
        object_insert_path(self, keys.into_iter().peekable(), value)
    }

    fn remove_path<'a, K>(&mut self, keys: K) -> Option<Self::Value>
    where K: IntoIterator<Item = &'a str>
    {
        fn object_remove_path<'a, K>(
            object: &mut JsonObject,
            mut keys: Peekable<K>,
        ) -> Option<json::Value>
        where K: Iterator<Item = &'a str>
        {
            if let Some(key) = keys.next() {
                match (object.get_mut(key), keys.peek()) {
                    (Some(json::Value::Object(subobj)), Some(_)) => {
                        object_remove_path(subobj, keys)
                    },
                    (_, None) => object.remove(key),
                    (Some(_), Some(_)) => None,
                    (None, _) => None,
                }
            } else {
                None
            }
        }
        object_remove_path(self, keys.into_iter().peekable())
    }
}

#[allow(unused_variables)]
impl InternalRecord for yaml::Mapping {
    type Value = yaml::Value;
    const DELIM: char = '.';

    fn get_path<'a, K>(&self, keys: K) -> Option<&Self::Value>
    where K: IntoIterator<Item = &'a str>
    {
        fn mapping_get_path<'a, K>(
            mapping: &yaml::Mapping,
            mut keys: Peekable<K>,
        ) -> Option<&yaml::Value>
        where K: Iterator<Item = &'a str>
        {
            if let Some(key) = keys.next() {
                match (mapping.get(key), keys.peek()) {
                    (Some(yaml::Value::Mapping(submap)), Some(_)) => {
                        mapping_get_path(submap, keys)
                    },
                    (x, None) => x,
                    (Some(_), Some(_)) => None,
                    (None, _) => None,
                }
            } else {
                None
            }
        }
        mapping_get_path(self, keys.into_iter().peekable())
    }

    fn get_mut_path<'a, K>(&mut self, keys: K) -> Option<&mut Self::Value>
    where K: IntoIterator<Item = &'a str>
    {
        fn mapping_get_mut_path<'a, K>(
            mapping: &mut yaml::Mapping,
            mut keys: Peekable<K>,
        ) -> Option<&mut yaml::Value>
        where K: Iterator<Item = &'a str>
        {
            if let Some(key) = keys.next() {
                match (mapping.get_mut(key), keys.peek()) {
                    (Some(yaml::Value::Mapping(submap)), Some(_)) => {
                        mapping_get_mut_path(submap, keys)
                    },
                    (x, None) => x,
                    (Some(_), Some(_)) => None,
                    (None, _) => None,
                }
            } else {
                None
            }
        }
        mapping_get_mut_path(self, keys.into_iter().peekable())
    }

    /// This method panics if the user tries to `insert` a value at the end of a
    /// path containing a non-table value anywhere before its end.
    fn insert_path<'a, K, T>(&mut self, keys: K, value: T)
        -> Option<Self::Value>
    where
        K: IntoIterator<Item = &'a str>,
        T: Into<Self::Value>,
    {
        fn mapping_insert_path<'a, K, T>(
            mapping: &mut yaml::Mapping,
            mut keys: Peekable<K>,
            value: T,
        ) -> Option<yaml::Value>
        where
            K: Iterator<Item = &'a str>,
            T: Into<yaml::Value>,
        {
            if let Some(key) = keys.next() {
                match (mapping.get_mut(key), keys.peek()) {
                    (Some(yaml::Value::Mapping(submap)), Some(_)) => {
                        mapping_insert_path(submap, keys, value)
                    },
                    (None, Some(_)) => {
                        mapping.insert(
                            key.into(), yaml::Mapping::new().into());
                        mapping_insert_path(
                            mapping.get_mut(key).unwrap()
                                .as_mapping_mut().unwrap(),
                            keys,
                            value,
                        )
                    },
                    (Some(x), None) => {
                        Some(std::mem::replace(x, value.into()))
                    },
                    (None, None) => {
                        mapping.insert(key.into(), value.into())
                    },
                    (Some(_), Some(_)) => {
                        panic!(
                            "InternalRecord for yaml::Map: encountered \
                            non-empty values in the middle of a path insertion"
                        );
                    },
                }
            } else {
                None
            }
        }
        mapping_insert_path(self, keys.into_iter().peekable(), value)
    }

    fn remove_path<'a, K>(&mut self, keys: K) -> Option<Self::Value>
    where K: IntoIterator<Item = &'a str>
    {
        fn mapping_remove_path<'a, K>(
            mapping: &mut yaml::Mapping,
            mut keys: Peekable<K>,
        ) -> Option<yaml::Value>
        where K: Iterator<Item = &'a str>
        {
            if let Some(key) = keys.next() {
                match (mapping.get_mut(key), keys.peek()) {
                    (Some(yaml::Value::Mapping(submap)), Some(_)) => {
                        mapping_remove_path(submap, keys)
                    },
                    (_, None) => mapping.remove(key),
                    (Some(_), Some(_)) => None,
                    (None, _) => None,
                }
            } else {
                None
            }
        }
        mapping_remove_path(self, keys.into_iter().peekable())
    }
}

/// An immutable wrapper around a verified config `T` holding values of type
/// `V`.
///
/// To mutate, first convert to a [`ConfigUnver`].
///
/// See also [`ConfigVerifier`].
#[derive(Clone, Debug)]
pub struct Config<T, V>
where T: InternalRecord<Value = V>
{
    data: T,
}

impl<T, V> AsRef<T> for Config<T, V>
where T: InternalRecord<Value = V>
{
    fn as_ref(&self) -> &T { &self.data }
}

impl<T, V> Config<T, V>
where T: InternalRecord<Value = V>
{
    /// Load config values from a file.
    pub fn from_file<P, C>(infile: P, verifier: &C) -> ConfigResult<Self>
    where
        V: FromStr,
        P: AsRef<Path>,
        C: ConfigVerifier<V, Record = Self, Error = ConfigError>,
    {
        let infile_str: String = infile.as_ref().display().to_string();
        let value: V
            = fs::read_to_string(infile)
            .map_err(|_| ConfigError::FileRead(infile_str.clone()))?
            .parse()
            .map_err(|_| ConfigError::FileParse(infile_str.clone()))?;
        verifier.verify_into(value)
    }

    /// Load config values from a string.
    pub fn from_str<C, U>(s: &str, verifier: &C) -> ConfigResult<Self>
    where
        U: FromStr,
        C: ConfigVerifier<U, Record = Self, Error = ConfigError>,
    {
        let value: U = s.parse().map_err(|_| ConfigError::StrParse)?;
        verifier.verify_into(value)
    }

    /// Wrap an underlying record type.
    pub fn from_data<C, U>(data: T, verifier: &C) -> ConfigResult<Self>
    where
        C: ConfigVerifier<U, Record = Self, Error = ConfigError>,
        T: Into<U>,
    {
        verifier.verify_into(data.into())
    }

    /// Convert to an unverified wrapper.
    pub fn into_unver(self) -> ConfigUnver<T, V> {
        let Self { data } = self;
        ConfigUnver { data }
    }

    /// Convert from an unverified wrapper.
    pub fn from_unver<C, U>(unver: ConfigUnver<T, V>, verifier: &C)
        -> ConfigResult<Self>
    where
        C: ConfigVerifier<U, Record = Self, Error = ConfigError>,
        T: Into<U>,
    {
        unver.into_ver(verifier)
    }

    /// Unwrap the underlying record.
    pub fn into_raw(self) -> T { self.data }

    /// Get a reference to the value at the end of a key path.
    pub fn get(&self, keypath: &str) -> Option<&V> {
        self.data.get(keypath)
    }

    /// Get a reference to the value at the end of a key path, returning `Err`
    /// if it doesn't exist.
    pub fn get_ok(&self, keypath: &str) -> ConfigResult<&V> {
        self.get(keypath)
            .ok_or(ConfigError::MissingKey(keypath.to_string()))
    }
}

impl<T, V> Config<T, V>
where T: InternalRecord<Value = V> + Serialize
{
    /// Generate a TOML-formatted string from the underlying data.
    pub fn as_toml_string(&self) -> ConfigResult<String> {
        Ok(toml::to_string(&self.data)?)
    }

    /// Generate a TOML-formatted "pretty" string from the underlying data.
    pub fn as_toml_string_pretty(&self) -> ConfigResult<String> {
        Ok(toml::to_string_pretty(&self.data)?)
    }

    /// Write the underlying data to a TOML-formatted file.
    pub fn write_toml<P>(&self, outfile: P, append: bool)
        -> ConfigResult<&Self>
    where P: AsRef<Path>
    {
        write_str_to_file(outfile, &self.as_toml_string()?, append)?;
        Ok(self)
    }

    /// Write the underlying data to a "pretty" TOML-formatted file.
    pub fn write_toml_pretty<P>(&self, outfile: P, append: bool)
        -> ConfigResult<&Self>
    where P: AsRef<Path>
    {
        write_str_to_file(outfile, &self.as_toml_string_pretty()?, append)?;
        Ok(self)
    }

    /// Generate a JSON-formatted string from the underlying data.
    pub fn as_json_string(&self) -> ConfigResult<String> {
        Ok(json::to_string(&self.data)?)
    }

    /// Generate a JSON-formatted "pretty" string from the underlying data.
    pub fn as_json_string_pretty(&self) -> ConfigResult<String> {
        Ok(json::to_string_pretty(&self.data)?)
    }

    /// Write the underlying data to a JSON-formatted file.
    pub fn write_json<P>(&self, outfile: P, append: bool)
        -> ConfigResult<&Self>
    where P: AsRef<Path>
    {
        write_str_to_file(outfile, &self.as_json_string()?, append)?;
        Ok(self)
    }

    /// Write the underlying data to a "pretty" JSON-formatted file.
    pub fn write_json_pretty<P>(&self, outfile: P, append: bool)
        -> ConfigResult<&Self>
    where P: AsRef<Path>
    {
        write_str_to_file(outfile, &self.as_json_string_pretty()?, append)?;
        Ok(self)
    }

    /// Generate a YAML-formatted string from the underlying data.
    pub fn as_yaml_string(&self) -> ConfigResult<String> {
        Ok(yaml::to_string(&self.data)?)
    }

    /// Write the underlying data to a YAML-formatted file.
    pub fn write_yaml<P>(&self, outfile: P, append: bool)
        -> ConfigResult<&Self>
    where P: AsRef<Path>
    {
        write_str_to_file(outfile, &self.as_yaml_string()?, append)?;
        Ok(self)
    }
}

pub type TomlConfig = Config<toml::Table, toml::Value>;
pub type JsonConfig = Config<JsonObject, json::Value>;
pub type YamlConfig = Config<yaml::Mapping, yaml::Value>;

impl TomlConfig {
    /// Convert the value at the end of a key path to a new type if it exists.
    pub fn get_into<'de, U>(&self, keypath: &str) -> Option<ConfigResult<U>>
    where U: Deserialize<'de>
    {
        self.data.get(keypath)
            .map(|v| {
                v.clone().try_into()
                    .map_err(|_| {
                        ConfigError::FailedTypeConversion(keypath.to_string())
                    })
            })
    }

    /// Convert the value at the end of a key path into a new type, returning
    /// `Err` if it doesn't exist or the conversion fails.
    pub fn get_into_ok<'de, U>(&self, keypath: &str) -> ConfigResult<U>
    where U: Deserialize<'de>
    {
        self.get_ok(keypath)
            .and_then(|v| {
                v.clone().try_into()
                    .map_err(|_| {
                        ConfigError::FailedTypeConversion(keypath.to_string())
                    })
            })
    }
}

impl<T, V> From<Config<T, V>> for ConfigUnver<T, V>
where T: InternalRecord<Value = V>
{
    fn from(config: Config<T, V>) -> Self { config.into_unver() }
}

/// A wrapper around an uverified config `T` holding values of type
/// `V`.
///
/// See also [`ConfigVerifier`] and [`Config`].
#[derive(Clone, Debug, Default)]
pub struct ConfigUnver<T, V>
where T: InternalRecord<Value = V>
{
    data: T,
}

impl<T, V> AsRef<T> for ConfigUnver<T, V>
where T: InternalRecord<Value = V>
{
    fn as_ref(&self) -> &T { &self.data }
}

impl<T, V> AsMut<T> for ConfigUnver<T, V>
where T: InternalRecord<Value = V>
{
    fn as_mut(&mut self) -> &mut T { &mut self.data }
}

impl<T, V> FromStr for ConfigUnver<T, V>
where T: FromStr + InternalRecord<Value = V>
{
    type Err = ConfigError;

    fn from_str(s: &str) -> ConfigResult<Self> {
        Ok(Self { data: s.parse().map_err(|_| ConfigError::StrParse)? })
    }
}

impl<T, V> ConfigUnver<T, V>
where T: InternalRecord<Value = V>
{
    /// Create a new, [`Default`] config.
    pub fn new() -> Self
    where T: Default
    {
        Self { data: Default::default() }
    }

    /// Load config values from a file.
    pub fn from_file<P>(infile: P) -> ConfigResult<Self>
    where
        T: FromStr,
        P: AsRef<Path>,
    {
        let infile_str: String = infile.as_ref().display().to_string();
        let data: T
            = fs::read_to_string(infile)
            .map_err(|_| ConfigError::FileRead(infile_str.clone()))?
            .parse()
            .map_err(|_| ConfigError::FileParse(infile_str.clone()))?;
        Ok(Self { data })
    }

    /// Wrap an underlying record type.
    pub fn from_data(data: T) -> Self { Self { data } }

    /// Convert from a verified wrapper.
    pub fn from_ver(ver: Config<T, V>) -> Self { ver.into() }

    /// Convert to a verified wrapper.
    pub fn into_ver<C, U>(self, verifier: &C) -> ConfigResult<Config<T, V>>
    where
        C: ConfigVerifier<U, Record = Config<T, V>, Error = ConfigError>,
        T: Into<U>,
    {
        verifier.verify_into(self.data.into())
    }

    /// Unwrap the underlying record.
    pub fn into_raw(self) -> T { self.data }

    /// Get a reference to the value at the end of a key path.
    pub fn get(&self, keypath: &str) -> Option<&V> {
        self.data.get(keypath)
    }

    /// Get a reference to the value at the end of a key path, returning `Err`
    /// if it doesn't exist.
    pub fn get_ok(&self, keypath: &str) -> ConfigResult<&V> {
        self.get(keypath)
            .ok_or(ConfigError::MissingKey(keypath.to_string()))
    }

    /// Get a mutable reference to the value at the end of a key path.
    pub fn get_mut(&mut self, keypath: &str) -> Option<&mut V> {
        self.data.get_mut(keypath)
    }

    /// Get a mutable reference to the value at the end of a key path, returning
    /// `Err` if it doesn't exist.
    pub fn get_mut_ok(&mut self, keypath: &str) -> ConfigResult<&mut V> {
        self.get_mut(keypath)
            .ok_or(ConfigError::MissingKey(keypath.to_string()))
    }

    /// Insert a value at the end of a key path, returning the previous value at
    /// that location if it existed.
    pub fn insert<U>(&mut self, keypath: &str, value: U) -> Option<V>
    where U: Into<V>
    {
        self.data.insert(keypath, value)
    }

    /// Insert a value at the end of a key path, returning the previous value or
    /// `Err` if it didn't exist.
    pub fn insert_ok<U>(&mut self, keypath: &str, value: U) -> ConfigResult<V>
    where U: Into<V>
    {
        self.insert(keypath, value)
            .ok_or(ConfigError::MissingKey(keypath.to_string()))
    }

    /// Insert a value at the end of a key path, converting the previous value
    /// at that location to a new type if it existed.
    pub fn insert_into<W, U>(&mut self, keypath: &str, value: W)
        -> Option<ConfigResult<U>>
    where
        W: Into<V>,
        V: TryInto<U>,
    {
        self.insert(keypath, value)
            .map(|v| {
                v.try_into()
                    .map_err(|_| {
                        ConfigError::FailedTypeConversion(keypath.to_string())
                    })
            })
    }

    /// Insert a value at the end of a key path, converting the previous value
    /// at that location to a new type or returning `Err` if it didn't exist or
    /// the type conversion failed.
    pub fn insert_into_ok<W, U>(&mut self, keypath: &str, value: W)
        -> ConfigResult<U>
    where
        W: Into<V>,
        V: TryInto<U>,
    {
        self.insert_ok(keypath, value)
            .and_then(|v| {
                v.try_into()
                    .map_err(|_| {
                        ConfigError::FailedTypeConversion(keypath.to_string())
                    })
            })
    }

    /// Remove a value at the end of a key path if it existed.
    pub fn remove(&mut self, keypath: &str) -> Option<V> {
        self.data.remove(keypath)
    }

    /// Remove a value at the end of a key path, returning `Err` if it didn't
    /// exist.
    pub fn remove_ok(&mut self, keypath: &str) -> ConfigResult<V> {
        self.remove(keypath)
            .ok_or(ConfigError::MissingKey(keypath.to_string()))
    }

    /// Remove a value at the end of a key path, converting it to a new type if
    /// it existed.
    pub fn remove_into<U>(&mut self, keypath: &str) -> Option<ConfigResult<U>>
    where V: TryInto<U>
    {
        self.remove(keypath)
            .map(|v| {
                v.try_into()
                    .map_err(|_| {
                        ConfigError::FailedTypeConversion(keypath.to_string())
                    })
            })
    }

    /// Remove a value at the end of a key path, converting it to a new type or
    /// returning `Err` if it didn't exist or the type conversion failed.
    pub fn remove_into_ok<U>(&mut self, keypath: &str) -> ConfigResult<U>
    where V: TryInto<U>
    {
        self.remove_ok(keypath)
            .and_then(|v| {
                v.try_into()
                    .map_err(|_| {
                        ConfigError::FailedTypeConversion(keypath.to_string())
                    })
            })
    }
}

impl<T, V> ConfigUnver<T, V>
where T: InternalRecord<Value = V> + Serialize
{
    /// Generate a TOML-formatted string from the underlying data.
    pub fn as_toml_string(&self) -> ConfigResult<String> {
        Ok(toml::to_string(&self.data)?)
    }

    /// Generate a TOML-foratted "pretty" string from the underlying data.
    pub fn as_toml_string_pretty(&self) -> ConfigResult<String> {
        Ok(toml::to_string_pretty(&self.data)?)
    }

    /// Write the underlying data to a TOML-formatted file.
    pub fn write_toml<P>(&self, outfile: P, append: bool)
        -> ConfigResult<&Self>
    where P: AsRef<Path>
    {
        write_str_to_file(outfile, &self.as_toml_string()?, append)?;
        Ok(self)
    }

    /// Write the underlying data to a "pretty" TOML-formatted file.
    pub fn write_toml_pretty<P>(&self, outfile: P, append: bool)
        -> ConfigResult<&Self>
    where P: AsRef<Path>
    {
        write_str_to_file(outfile, &self.as_toml_string_pretty()?, append)?;
        Ok(self)
    }

    /// Generate a JSON-formatted string from the underlying data.
    pub fn as_json_string(&self) -> ConfigResult<String> {
        Ok(json::to_string(&self.data)?)
    }

    /// Generate a JSON-formatted "pretty" string from the underlying data.
    pub fn as_json_string_pretty(&self) -> ConfigResult<String> {
        Ok(json::to_string_pretty(&self.data)?)
    }

    /// Write the underlying data to a JSON-formatted file.
    pub fn write_json<P>(&self, outfile: P, append: bool)
        -> ConfigResult<&Self>
    where P: AsRef<Path>
    {
        write_str_to_file(outfile, &self.as_json_string()?, append)?;
        Ok(self)
    }

    /// Write the underlying data to a "pretty" JSON-formatted file.
    pub fn write_json_pretty<P>(&self, outfile: P, append: bool)
        -> ConfigResult<&Self>
    where P: AsRef<Path>
    {
        write_str_to_file(outfile, &self.as_json_string_pretty()?, append)?;
        Ok(self)
    }

    /// Generate a YAML-formatted string from the underlying data.
    pub fn as_yaml_string(&self) -> ConfigResult<String> {
        Ok(yaml::to_string(&self.data)?)
    }

    /// Write the underlying data to a YAML-formatted file.
    pub fn write_yaml<P>(&self, outfile: P, append: bool)
        -> ConfigResult<&Self>
    where P: AsRef<Path>
    {
        write_str_to_file(outfile, &self.as_yaml_string()?, append)?;
        Ok(self)
    }
}

pub type TomlConfigUnver = ConfigUnver<toml::Table, toml::Value>;
pub type JsonConfigUnver = ConfigUnver<JsonObject, json::Value>;
pub type YamlConfigUnver = ConfigUnver<yaml::Mapping, yaml::Value>;

impl TomlConfigUnver {
    /// Convert the value at the end of a key path to a new type if it exists.
    pub fn get_into<'de, U>(&self, keypath: &str) -> Option<ConfigResult<U>>
    where U: Deserialize<'de>
    {
        self.data.get(keypath)
            .map(|v| {
                v.clone().try_into()
                    .map_err(|_| {
                        ConfigError::FailedTypeConversion(keypath.to_string())
                    })
            })
    }

    /// Convert the value at the end of a key path into a new type, returning
    /// `Err` if it doesn't exist or the conversion fails.
    pub fn get_into_ok<'de, U>(&self, keypath: &str) -> ConfigResult<U>
    where U: Deserialize<'de>
    {
        self.get_ok(keypath)
            .and_then(|v| {
                v.clone().try_into()
                    .map_err(|_| {
                        ConfigError::FailedTypeConversion(keypath.to_string())
                    })
            })
    }
}

fn write_str_to_file<P>(outfile: P, s: &str, append: bool) -> ConfigResult<()>
where P: AsRef<Path>
{
    let outfile_string = outfile.as_ref().display().to_string();
    let mut out
        = fs::OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(!append)
        .append(append)
        .open(outfile)
        .map_err(|e| {
            ConfigError::FileOpen(outfile_string.clone(), e.to_string())
        })?;
    write!(&mut out, "{}", s)
        .map_err(|e| {
            ConfigError::FileWrite(outfile_string.clone(), e.to_string())
        })?;
    Ok(())
}

