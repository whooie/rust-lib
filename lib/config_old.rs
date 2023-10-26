//! Provides definitions for loading and verifying values from TOML-formatted
//! config files.

use std::{
    collections::{
        HashMap,
        HashSet,
    },
    fmt,
    fs,
    io::Write,
    iter::Peekable,
    ops::{
        Deref,
        DerefMut,
    },
    path::Path,
};
use toml::{
    Value,
    Table,
};
use serde::Deserialize;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ConfigError {
    #[error("config: invalid type for key '{0}':\n\texpected type <{1}> but got value '{2}'")]
    InvalidType(String, String, String),

    #[error("config: invalid value for key '{0}':\n\texpected value to satisfy '{1}' but got '{2}'")]
    InvalidValue(String, String, String),

    #[error("config: encountered imcompatible structure")]
    IncompatibleStructure,

    #[error("config: missing key '{0}'")]
    MissingKey(String),

    #[error("config: key path too long at key '{0}'")]
    KeyPathTooLong(String),

    #[error("config: failed to convert type of value {0}")]
    FailedTypeConversion(String),

    #[error("config: couldn't read file '{0}'")]
    FileRead(String),

    #[error("config: couldn't parse file '{0}'")]
    FileParse(String),

    #[error("config: couldn't open file '{0}':\n\t{1}")]
    FileOpen(String, String),

    #[error("config: couldn't write to file '{0}':\n\t{1}")]
    FileWrite(String, String),

    #[error("config: serialization error '{0}'")]
    SerializationError(#[from] toml::ser::Error),
}
pub type ConfigResult<T> = Result<T, ConfigError>;

/// Config value type specification via the TOML format.
///
/// See [`toml::Value`]
#[derive(Clone, Debug)]
pub enum TypeVer {
    /// [`toml::Value::Boolean`]
    Bool,
    /// [`toml::Value::Integer`]
    Int,
    /// [`toml::Value::Float`]
    Float,
    /// [`toml::Value::Datetime`]
    Datetime,
    /// [`toml::Value::Array`]
    Array,
    /// [`toml::Value::Array`] holding only specified types, optionally in
    /// specific positions.
    TypedArray {
        types: Vec<TypeVer>,
        finite: bool,
    },
    /// [`toml::Value::String`]
    Str,
    /// [`toml::Value::Table`].
    ///
    /// See [`ConfigSpecItem`] to recurse into a full specification for the
    /// table.
    Table,
    /// Any single TOML type.
    Any,
    /// Any TOML type, including a table.
    AnyTable,
}

impl TypeVer {
    /// Return `true` if `value` matches the type specification, `false`
    /// otherwise.
    pub fn verify(&self, value: &Value) -> bool {
        return match (self, value) {
            (Self::Bool, Value::Boolean(_)) => true,
            (Self::Int, Value::Integer(_)) => true,
            (Self::Float, Value::Float(_)) => true,
            (Self::Datetime, Value::Datetime(_)) => true,
            (Self::TypedArray { types, finite }, Value::Array(a)) => {
                if *finite {
                    types.len() == a.len()
                        && types.iter().zip(a.iter())
                            .all(|(tyk, ak)| tyk.verify(ak))
                } else {
                    a.iter()
                        .all(|ak| types.iter().any(|tyk| tyk.verify(ak)))
                }
            },
            (Self::Str, Value::String(_)) => true,
            (Self::Table, Value::Table(_)) => true,
            (Self::Any, Value::Boolean(_))
                | (Self::Any, Value::Integer(_))
                | (Self::Any, Value::Float(_))
                | (Self::Any, Value::Datetime(_))
                | (Self::Any, Value::Array(_))
                => true,
            (Self::AnyTable, _) => true,
            _ => false,
        };
    }

    /// Return `Ok(value)` if `value` matches the type specification, `Err(err)`
    /// otherwise.
    pub fn verify_ok_or(&self, value: Value, err: ConfigError)
        -> ConfigResult<Value>
    {
        return self.verify(&value).then_some(value).ok_or(err);
    }
}

impl fmt::Display for TypeVer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        return match self {
            Self::Bool => write!(f, "Bool"),
            Self::Int => write!(f, "Int"),
            Self::Float => write!(f, "Float"),
            Self::Datetime => write!(f, "Datetime"),
            Self::Array => write!(f, "Array"),
            Self::TypedArray { types, finite } => {
                let n: usize = types.len();
                write!(f, "TypedArray([")?;
                for (k, tyk) in types.iter().enumerate() {
                    tyk.fmt(f)?;
                    if k < n - 1 { write!(f, ", ")?; }
                }
                write!(f, ", finite={})", finite)?;
                Ok(())
            },
            Self::Str => write!(f, "Str"),
            Self::Table => write!(f, "Table"),
            Self::Any => write!(f, "Any"),
            Self::AnyTable => write!(f, "AnyTable"),
        };
    }
}

/// Config specification to bound possible values found in a config file.
#[derive(Clone, Debug)]
pub enum ValueVer {
    IntRange {
        min: i64,
        max: i64,
        incl_start: bool,
        incl_end: bool,
    },
    FloatRange {
        min: f64,
        max: f64,
        incl_start: bool,
        incl_end: bool,
    },
    StrColl(HashSet<String>),
}

impl ValueVer {
    /// Return `true` if `value` matches the specification, `false` otherwise.
    pub fn verify(&self, value: &Value) -> bool {
        return match (self, value) {
            (
                Self::IntRange { min, max, incl_start, incl_end },
                Value::Integer(i),
            ) => {
                let in_start: bool
                    = if *incl_start { *i >= *min } else { *i > *min };
                let in_end: bool
                    = if *incl_end { *i <= *max } else { *i < *max };
                in_start && in_end
            },
            (
                Self::FloatRange { min, max, incl_start, incl_end },
                Value::Float(f),
            ) => {
                let in_start: bool
                    = if *incl_start { *f >= *min } else { *f > *min };
                let in_end: bool
                    = if *incl_end { *f <= *max } else { *f < *max };
                in_start && in_end
            },
            (Self::StrColl(strs), Value::String(s)) => strs.contains(s),
            _ => false,
        };
    }

    /// Return `Ok(value)` if `value` matches the specification, `Err(err)`
    /// otherwise.
    pub fn verify_ok_or(&self, value: Value, err: ConfigError)
        -> ConfigResult<Value>
    {
        return self.verify(&value).then_some(value).ok_or(err);
    }
}

impl fmt::Display for ValueVer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        return match self {
            Self::IntRange { min, max, incl_start, incl_end } => {
                write!(f, "IntRange{}", if *incl_start { "[" } else { "(" })?;
                min.fmt(f)?;
                write!(f, ", ")?;
                max.fmt(f)?;
                write!(f, "{}", if *incl_end { "]" } else { ")" })?;
                Ok(())
            },
            Self::FloatRange { min, max, incl_start, incl_end } => {
                write!(f, "FloatRange{}", if *incl_start { "[" } else { "(" })?;
                min.fmt(f)?;
                write!(f, ", ")?;
                max.fmt(f)?;
                write!(f, "{}", if *incl_end { "]" } else { ")" })?;
                Ok(())
            },
            Self::StrColl(strs) => {
                let n: usize = strs.len();
                write!(f, "StrCollection{{")?;
                for (k, sk) in strs.iter().enumerate() {
                    sk.fmt(f)?;
                    if k < n - 1 { write!(f, ", ")?; }
                }
                write!(f, "}}")?;
                Ok(())
            },
        };
    }
}

/// Holds either a type or value specification.
#[derive(Clone, Debug)]
pub enum Verifier {
    TypeVer(TypeVer),
    ValueVer(ValueVer),
}

impl fmt::Display for Verifier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        return match self {
            Self::TypeVer(type_ver) => type_ver.fmt(f),
            Self::ValueVer(value_ver) => value_ver.fmt(f),
        };
    }
}

/// An item in a config specification, either a type/value specification for
/// single values or a set of specifications for a sub-table.
#[derive(Clone, Debug)]
pub enum ConfigSpecItem {
    Value(Verifier),
    Table(ConfigSpec),
}

/// Shorthand for creating [`ConfigSpecItem`]s.
#[macro_export]
macro_rules! spec {
    ( Bool ) => {
        $crate::config::ConfigSpecItem::Value(
            $crate::config::Verifier::TypeVer(
                $crate::config::TypeVer::Bool
            )
        )
    };
    ( Int ) => {
        $crate::config::ConfigSpecItem::Value(
            $crate::config::Verifier::TypeVer(
                $crate::config::TypeVer::Int
            )
        )
    };
    ( Float ) => {
        $crate::config::ConfigSpecItem::Value(
            $crate::config::Verifier::TypeVer(
                $crate::config::TypeVer::Float
            )
        )
    };
    ( Datetime ) => {
        $crate::config::ConfigSpecItem::Value(
            $crate::config::Verifier::TypeVer(
                $crate::config::TypeVer::Datetime
            )
        )
    };
    ( Array ) => {
        $crate::config::ConfigSpecItem::Value(
            $crate::config::Verifier::TypeVer(
                $crate::config::TypeVer::Array
            )
        )
    };
    ( TypedArray { [ $( $t:expr ),* $(,)? ], finite: $f:expr } ) => {
        $crate::config::ConfigSpecItem::Value(
            $crate::config::Verifier::TypeVer(
                $crate::config::TypeVer::TypedArray {
                    types: vec![ $( $t ),* ], finite: $f
                }
            )
        )
    };
    ( Str ) => {
        $crate::config::ConfigSpecItem::Value(
            $crate::config::Verifier::TypeVer(
                $crate::config::TypeVer::Str
            )
        )
    };
    ( Table ) => {
        $crate::config::ConfigSpecItem::Value(
            $crate::config::Verifier::TypeVer(
                $crate::config::TypeVer::Table
            )
        )
    };
    ( Any ) => {
        $crate::config::ConfigSpecItem::Value(
            $crate::config::Verifier::TypeVer(
                $crate::config::TypeVer::Any
            )
        )
    };
    ( AnyTable ) => {
        $crate::config::ConfigSpecItem::Value(
            $crate::config::Verifier::TypeVer(
                $crate::config::TypeVer::AnyTable
            )
        )
    };
    ( IntRange { ($min:expr, $max:expr) } ) => {
        $crate::config::ConfigSpecItem::Value(
            $crate::config::Verifier::ValueVer(
                $crate::config::ValueVer::IntRange {
                    min: $min,
                    max: $max,
                    incl_start: false,
                    incl_end: false,
                }
            )
        )
    };
    ( IntRange { ($min:expr, $max:expr)= } ) => {
        $crate::config::ConfigSpecItem::Value(
            $crate::config::Verifier::ValueVer(
                $crate::config::ValueVer::IntRange {
                    min: $min,
                    max: $max,
                    incl_start: false,
                    incl_end: true,
                }
            )
        )
    };
    ( IntRange { =($min:expr, $max:expr) } ) => {
        $crate::config::ConfigSpecItem::Value(
            $crate::config::Verifier::ValueVer(
                $crate::config::ValueVer::IntRange {
                    min: $min,
                    max: $max,
                    incl_start: true,
                    incl_end: false,
                }
            )
        )
    };
    ( IntRange { =($min:expr, $max:expr)= } ) => {
        $crate::config::ConfigSpecItem::Value(
            $crate::config::Verifier::ValueVer(
                $crate::config::ValueVer::IntRange {
                    min: $min,
                    max: $max,
                    incl_start: true,
                    incl_end: true,
                }
            )
        )
    };
    ( FloatRange { ($min:expr, $max:expr) } ) => {
        $crate::config::ConfigSpecItem::Value(
            $crate::config::Verifier::ValueVer(
                $crate::config::ValueVer::FloatRange {
                    min: $min,
                    max: $max,
                    incl_start: false,
                    incl_end: false,
                }
            )
        )
    };
    ( FloatRange { ($min:expr, $max:expr)= } ) => {
        $crate::config::ConfigSpecItem::Value(
            $crate::config::Verifier::ValueVer(
                $crate::config::ValueVer::FloatRange {
                    min: $min,
                    max: $max,
                    incl_start: false,
                    incl_end: true,
                }
            )
        )
    };
    ( FloatRange { =($min:expr, $max:expr) } ) => {
        $crate::config::ConfigSpecItem::Value(
            $crate::config::Verifier::ValueVer(
                $crate::config::ValueVer::FloatRange {
                    min: $min,
                    max: $max,
                    incl_start: true,
                    incl_end: false,
                }
            )
        )
    };
    ( FloatRange { =($min:expr, $max:expr)= } ) => {
        $crate::config::ConfigSpecItem::Value(
            $crate::config::Verifier::ValueVer(
                $crate::config::ValueVer::FloatRange {
                    min: $min,
                    max: $max,
                    incl_start: true,
                    incl_end: true,
                }
            )
        )
    };
    ( Table { $( $key:literal => $val:expr ),* $(,)? } ) => {
        $crate::config::ConfigSpecItem::Table(
            $crate::config::ConfigSpec::from_iter([
                $(
                    ($key.to_string(), $val)
                ),*
            ])
        )
    };
    ( StrColl { $( $s:expr ),* $(,)? } ) => {
        $crate::config::ConfigSpecItem::Value(
            $crate::config::Verifier::ValueVer(
                $crate::config::ValueVer::StrColl(
                    std::collections::HashSet::from_iter([
                        $( $s.to_string() ),*
                    ])
                )
            )
        )
    };
}

/// Sugared [`std::collections::HashMap`] representing a specification for the
/// values and structure of a TOML-formatted config file.
#[derive(Clone, Debug)]
pub struct ConfigSpec {
    spec: HashMap<String, ConfigSpecItem>
}

impl AsRef<HashMap<String, ConfigSpecItem>> for ConfigSpec {
    fn as_ref(&self) -> &HashMap<String, ConfigSpecItem> { &self.spec }
}

impl AsMut<HashMap<String, ConfigSpecItem>> for ConfigSpec {
    fn as_mut(&mut self) -> &mut HashMap<String, ConfigSpecItem> {
        return &mut self.spec;
    }
}

impl Deref for ConfigSpec {
    type Target = HashMap<String, ConfigSpecItem>;

    fn deref(&self) -> &Self::Target { &self.spec }
}

impl DerefMut for ConfigSpec {
    fn deref_mut(&mut self) -> &mut Self::Target {
        return &mut self.spec;
    }
}

impl From<HashMap<String, ConfigSpecItem>> for ConfigSpec {
    fn from(spec: HashMap<String, ConfigSpecItem>) -> Self { Self { spec } }
}

impl FromIterator<(String, ConfigSpecItem)> for ConfigSpec {
    fn from_iter<I>(iter: I) -> Self
    where I: IntoIterator<Item = (String, ConfigSpecItem)>
    {
        return Self { spec: iter.into_iter().collect() };
    }
}

/// Create a [`ConfigSpec`].
///
/// Expects `str` literals for keys and [`ConfigSpecItem`]s for values. See also
/// [`spec`].
#[macro_export]
macro_rules! config_spec {
    ( $( $key:literal => $val:expr ),* $(,)? ) => {
        $crate::config::ConfigSpec::from_iter([
            $(
                ($key.to_string(), $val)
            ),*
        ])
    }
}

impl ConfigSpec {
    pub fn from_spec(spec: HashMap<String, ConfigSpecItem>) -> Self {
        return Self { spec };
    }

    fn verify_ok(&self, value: Value) -> ConfigResult<Value> {
        if let Value::Table(mut tab) = value {
            let mut data = Table::new();
            let mut val: Value;
            let mut valstr: String;
            for (key, spec_item) in self.spec.iter() {
                val = match (spec_item, tab.remove(key)) {
                    (ConfigSpecItem::Table(spec_table), Some(v)) => {
                        spec_table.verify_ok(v)
                    },
                    (ConfigSpecItem::Value(verifier), Some(v)) => {
                        match verifier {
                            Verifier::TypeVer(type_ver) => {
                                valstr = v.to_string();
                                type_ver.verify_ok_or(
                                    v,
                                    ConfigError::InvalidType(
                                        key.clone(),
                                        type_ver.to_string(),
                                        valstr,
                                    )
                                )
                            },
                            Verifier::ValueVer(value_ver) => {
                                valstr = v.to_string();
                                value_ver.verify_ok_or(
                                    v,
                                    ConfigError::InvalidValue(
                                        key.clone(),
                                        value_ver.to_string(),
                                        valstr,
                                    )
                                )
                            },
                        }
                    },
                    _ => Err(ConfigError::MissingKey(key.clone())),
                }?;
                data.insert(key.clone(), val);
            }
            return Ok(Value::Table(data));
        } else {
            return Err(ConfigError::IncompatibleStructure);
        }
    }

    /// Return `Ok` if `table` matches `self`, `Err` otherwise.
    ///
    /// The keys of `table` are filtered to contain only those in the
    /// specification.
    pub fn verify(&self, table: Table) -> ConfigResult<Config> {
        let data: Table
            = self.verify_ok(Value::Table(table))?
            .try_into()
            .unwrap();
        return Ok(Config { data });
    }
}

/// Sugared [`toml::Table`] holding verified configuration values.
#[derive(Clone, Debug)]
pub struct Config {
    data: Table
}

impl AsRef<Table> for Config {
    fn as_ref(&self) -> &Table { &self.data }
}

impl Deref for Config {
    type Target = Table;

    fn deref(&self) -> &Self::Target { &self.data }
}

impl Config {
    /// Convert `self` into a [`ConfigUnver`]. This is necessary for mutation.
    pub fn into_unver(self) -> ConfigUnver { ConfigUnver { data: self.data } }

    /// Serialize `self` as a TOML-formatted string.
    pub fn as_toml_string(&self) -> ConfigResult<String> {
        return Ok(toml::to_string(&self.data)?);
    }

    /// Serialize `self` as a TOML-formatted 'pretty' string.
    pub fn as_toml_string_pretty(&self) -> ConfigResult<String> {
        return Ok(toml::to_string_pretty(&self.data)?);
    }

    /// Serialize `self` as a TOML-formatted string and write it to a file.
    pub fn write_toml<P>(&self, outfile: P, create: bool, append: bool)
        -> ConfigResult<()>
    where P: AsRef<Path>
    {
        let outfile_string: String
            = outfile.as_ref().display().to_string();
        let mut out
            = fs::File::options()
            .write(true)
            .create(create)
            .truncate(!append)
            .append(append)
            .open(outfile)
            .map_err(|e| {
                ConfigError::FileOpen(outfile_string.clone(), e.to_string())
            })?;
        write!(&mut out, "{}", self.as_toml_string()?)
            .map_err(|e| {
                ConfigError::FileWrite(outfile_string.clone(), e.to_string())
            })?;
        return Ok(());
    }

    /// Serialize `self` as a TOML-formatted 'pretty' string and write it to a
    /// file.
    pub fn write_toml_pretty<P>(&self, outfile: P, create: bool, append: bool)
        -> ConfigResult<()>
    where P: AsRef<Path>
    {
        let outfile_string: String
            = outfile.as_ref().display().to_string();
        let mut out
            = fs::File::options()
            .write(true)
            .create(create)
            .truncate(!append)
            .append(append)
            .open(outfile)
            .map_err(|e| {
                ConfigError::FileOpen(outfile_string.clone(), e.to_string())
            })?;
        write!(&mut out, "{}", self.as_toml_string_pretty()?)
            .map_err(|e| {
                ConfigError::FileWrite(outfile_string.clone(), e.to_string())
            })?;
        return Ok(());
    }

    /// Create a new [`Config`] with verification against [`spec`].
    pub fn new(data: Table, spec: &ConfigSpec) -> ConfigResult<Self> {
        return spec.verify(data);
    }

    /// Read a TOML file into a new [`Config`] with verification against
    /// [`spec`].
    pub fn from_file<P>(infile: P, verify: &ConfigSpec) -> ConfigResult<Self>
    where P: AsRef<Path>
    {
        let infile_str: String = infile.as_ref().display().to_string();
        let table: Table
            = fs::read_to_string(infile)
            .map_err(|_| ConfigError::FileRead(infile_str.clone()))?
            .parse()
            .map_err(|_| ConfigError::FileParse(infile_str.clone()))?;
        return verify.verify(table);
    }

    fn table_get_path<'a, K>(table: &Table, mut keys: Peekable<K>)
        -> Option<&Value>
    where K: Iterator<Item = &'a str>
    {
        return if let Some(key) = keys.next() {
            match (table.get(key), keys.peek()) {
                (Some(Value::Table(tab)), Some(_)) => {
                    Self::table_get_path(tab, keys)
                },
                (x, None) => x,
                (Some(_), Some(_)) => None,
                (None, _) => None,
            }
        } else {
            None
        };
    }

    /// Access a key path in `self`, returning `Some` if the complete path
    /// exists, `None` otherwise.
    pub fn get_path<'a, K>(&self, keys: K) -> Option<&Value>
    where K: IntoIterator<Item = &'a str>
    {
        return Self::table_get_path(&self.data, keys.into_iter().peekable());
    }

    /// Access a key path in `self` where the individual keys in the path are
    /// separated by `'.'`. Returns `Some` if the complete path exists, `None`
    /// otherwise.
    pub fn get(&self, keys: &str) -> Option<&Value> {
        return self.get_path(keys.split('.'));
    }

    /// Access a key path in `self` and attempt to convert its type to `T`,
    /// returning `Some(T)` if the complete path exists and the type is
    /// convertible, `None` otherwise.
    pub fn get_path_into<'a, 'de, K, T>(&self, keys: K) -> Option<T>
    where
        T: Deserialize<'de>,
        K: IntoIterator<Item = &'a str>,
    {
        return match self.get_path(keys) {
            Some(x) => x.clone().try_into().ok(),
            None => None,
        };
    }

    /// Access a key path in `self`, where individual keys in the path are
    /// separated by `'.'`, and attempt to convert its type to `T`, returning
    /// `Some(T)` if the complete path exists and the type is convertible,
    /// `None` otherwise.
    pub fn get_into<'de, T>(&self, keys: &str) -> Option<T>
    where T: Deserialize<'de>
    {
        return match self.get(keys) {
            Some(x) => x.clone().try_into().ok(),
            None => None,
        };
    }

    fn table_get_path_ok<'a, K>(table: &Table, mut keys: Peekable<K>)
        -> ConfigResult<&Value>
    where K: Iterator<Item = &'a str>
    {
        return if let Some(key) = keys.next() {
            match (table.get(key), keys.peek()) {
                (Some(Value::Table(tab)), Some(_)) => {
                    Self::table_get_path_ok(tab, keys)
                },
                (x, None) => {
                    x.ok_or_else(|| ConfigError::MissingKey(key.to_string()))
                },
                (Some(_), Some(k))
                    => Err(ConfigError::KeyPathTooLong(k.to_string())),
                (None, _) => Err(ConfigError::MissingKey(key.to_string())),
            }
        } else {
            unreachable!()
        };
    }

    /// Access a key path in `self`, returning `Ok` if the complete path
    /// exists, `Err` otherwise.
    pub fn get_path_ok<'a, K>(&self, keys: K) -> ConfigResult<&Value>
    where K: IntoIterator<Item = &'a str>
    {
        return Self::table_get_path_ok(&self.data, keys.into_iter().peekable());
    }

    /// Access a key path in `self` where the individual keys in the path are
    /// separated by `'.'`. Returns `Ok` if the complete path exists, `Err`
    /// otherwise.
    pub fn get_ok(&self, keys: &str) -> ConfigResult<&Value> {
        return self.get_path_ok(keys.split('.'));
    }

    /// Access a key path in `self` and attempt to convert its type to `T`,
    /// returning `Ok(T)` if the complete path exists and the type is
    /// convertible, `Err` otherwise.
    pub fn get_path_ok_into<'a, 'de, K, T>(&self, keys: K) -> ConfigResult<T>
    where
        T: Deserialize<'de>,
        K: IntoIterator<Item = &'a str>,
    {
        return self.get_path_ok(keys)
            .and_then(|x| {
                x.clone()
                    .try_into()
                    .map_err(|_| {
                        ConfigError::FailedTypeConversion(x.to_string())
                    })
            });
    }

    /// Access a key path in `self`, where individual keys in the path are
    /// separated by `'.'`, and attempt to convert its type to `T`, returning
    /// `Ok(T)` if the complete path exists and the type is convertible, `Err`
    /// otherwise.
    pub fn get_ok_into<'de, T>(&self, keys: &str) -> ConfigResult<T>
    where T: Deserialize<'de>
    {
        return self.get_ok(keys)
            .and_then(|x| {
                x.clone()
                    .try_into()
                    .map_err(|_| {
                        ConfigError::FailedTypeConversion(x.to_string())
                    })
            });
    }

    /// Alias for [`Self::get_ok_into`].
    pub fn a<'de, T>(&self, keys: &str) -> ConfigResult<T>
    where T: Deserialize<'de>
    {
        return self.get_ok_into(keys);
    }

    /// Convert `self` to a bare [`Table`].
    pub fn into_table(self) -> Table { self.data }
}

impl From<Config> for Table {
    fn from(config: Config) -> Self { config.data }
}

impl From<Config> for ConfigUnver {
    fn from(config: Config) -> Self { Self { data: config.data } }
}

/// Sugared [`toml::Table`] holding unverified configuration values.
#[derive(Clone, Debug)]
pub struct ConfigUnver {
    data: Table
}

impl AsRef<Table> for ConfigUnver {
    fn as_ref(&self) -> &Table { &self.data }
}

impl AsMut<Table> for ConfigUnver {
    fn as_mut(&mut self) -> &mut Table { &mut self.data }
}

impl Deref for ConfigUnver {
    type Target = Table;

    fn deref(&self) -> &Self::Target { &self.data }
}

impl DerefMut for ConfigUnver {
    fn deref_mut(&mut self) -> &mut Self::Target { &mut self.data }
}

impl Default for ConfigUnver {
    fn default() -> Self { Self::new() }
}

impl ConfigUnver {
    /// Serialize `self` as a TOML-formatted string.
    pub fn as_toml_string(&self) -> ConfigResult<String> {
        return Ok(toml::to_string(&self.data)?);
    }

    /// Serialize `self` as a TOML-formatted 'pretty' string.
    pub fn as_toml_string_pretty(&self) -> ConfigResult<String> {
        return Ok(toml::to_string_pretty(&self.data)?);
    }

    /// Serialize `self` as a TOML-formatted string and write it to a file.
    pub fn write_toml<P>(&self, outfile: P, create: bool, append: bool)
        -> ConfigResult<()>
    where P: AsRef<Path>
    {
        let outfile_string: String
            = outfile.as_ref().display().to_string();
        let mut out
            = fs::File::options()
            .write(true)
            .create(create)
            .truncate(!append)
            .append(append)
            .open(outfile)
            .map_err(|e| {
                ConfigError::FileOpen(outfile_string.clone(), e.to_string())
            })?;
        write!(&mut out, "{}", self.as_toml_string()?)
            .map_err(|e| {
                ConfigError::FileWrite(outfile_string.clone(), e.to_string())
            })?;
        return Ok(());
    }

    /// Serialize `self` as a TOML-formatted 'pretty' string and write it to a
    /// file.
    pub fn write_toml_pretty<P>(&self, outfile: P, create: bool, append: bool)
        -> ConfigResult<()>
    where P: AsRef<Path>
    {
        let outfile_string: String
            = outfile.as_ref().display().to_string();
        let mut out
            = fs::File::options()
            .write(true)
            .create(create)
            .truncate(!append)
            .append(append)
            .open(outfile)
            .map_err(|e| {
                ConfigError::FileOpen(outfile_string.clone(), e.to_string())
            })?;
        write!(&mut out, "{}", self.as_toml_string_pretty()?)
            .map_err(|e| {
                ConfigError::FileWrite(outfile_string.clone(), e.to_string())
            })?;
        return Ok(());
    }

    /// Create a new, empty [`ConfigUnver`].
    pub fn new() -> Self { Self { data: Table::new() } }

    /// Convert to a [`Config`] by verifying against `spec`.
    pub fn into_verified(self, spec: &ConfigSpec) -> ConfigResult<Config> {
        return spec.verify(self.data);
    }

    /// Read a TOML file into a new [`ConfigUnver`]
    pub fn from_file<P>(infile: P) -> ConfigResult<Self>
    where P: AsRef<Path>
    {
        let infile_str: String = infile.as_ref().display().to_string();
        let table: Table
            = fs::read_to_string(infile)
            .map_err(|_| ConfigError::FileRead(infile_str.clone()))?
            .parse()
            .map_err(|_| ConfigError::FileParse(infile_str.clone()))?;
        return Ok(Self { data: table });
    }

    fn table_get_path<'a, K>(table: &Table, mut keys: Peekable<K>)
        -> Option<&Value>
    where K: Iterator<Item = &'a str>
    {
        return if let Some(key) = keys.next() {
            match (table.get(key), keys.peek()) {
                (Some(Value::Table(tab)), Some(_)) => {
                    Self::table_get_path(tab, keys)
                },
                (x, None) => x,
                (Some(_), Some(_)) => None,
                (None, _) => None,
            }
        } else {
            None
        };
    }

    /// Access a key path in `self`, returning `Some` if the complete path
    /// exists, `None` otherwise.
    pub fn get_path<'a, K>(&self, keys: K) -> Option<&Value>
    where K: IntoIterator<Item = &'a str>
    {
        return Self::table_get_path(&self.data, keys.into_iter().peekable());
    }

    /// Access a key path in `self` where the individual keys in the path are
    /// separated by `'.'`. Returns `Some` if the complete path exists, `None`
    /// otherwise.
    pub fn get(&self, keys: &str) -> Option<&Value> {
        return self.get_path(keys.split('.'));
    }

    /// Access a key path in `self` and attempt to convert its type to `T`,
    /// returning `Some(T)` if the complete path exists and the type is
    /// convertible, `None` otherwise.
    pub fn get_path_into<'a, 'de, K, T>(&self, keys: K) -> Option<T>
    where
        T: Deserialize<'de>,
        K: IntoIterator<Item = &'a str>,
    {
        return match self.get_path(keys) {
            Some(x) => x.clone().try_into().ok(),
            None => None,
        };
    }

    /// Access a key path in `self`, where individual keys in the path are
    /// separated by `'.'`, and attempt to convert its type to `T`, returning
    /// `Some(T)` if the complete path exists and the type is convertible,
    /// `None` otherwise.
    pub fn get_into<'de, T>(&self, keys: &str) -> Option<T>
    where T: Deserialize<'de>
    {
        return match self.get(keys) {
            Some(x) => x.clone().try_into().ok(),
            None => None,
        };
    }

    fn table_get_path_ok<'a, K>(table: &Table, mut keys: Peekable<K>)
        -> ConfigResult<&Value>
    where K: Iterator<Item = &'a str>
    {
        return if let Some(key) = keys.next() {
            match (table.get(key), keys.peek()) {
                (Some(Value::Table(tab)), Some(_)) => {
                    Self::table_get_path_ok(tab, keys)
                },
                (x, None) => {
                    x.ok_or_else(|| ConfigError::MissingKey(key.to_string()))
                },
                (Some(_), Some(k))
                    => Err(ConfigError::KeyPathTooLong(k.to_string())),
                (None, _) => Err(ConfigError::MissingKey(key.to_string())),
            }
        } else {
            unreachable!()
        };
    }

    /// Access a key path in `self`, returning `Ok` if the complete path
    /// exists, `Err` otherwise.
    pub fn get_path_ok<'a, K>(&self, keys: K) -> ConfigResult<&Value>
    where K: IntoIterator<Item = &'a str>
    {
        return Self::table_get_path_ok(&self.data, keys.into_iter().peekable());
    }

    /// Access a key path in `self` where the individual keys in the path are
    /// separated by `'.'`. Returns `Ok` if the complete path exists, `Err`
    /// otherwise.
    pub fn get_ok(&self, keys: &str) -> ConfigResult<&Value> {
        return self.get_path_ok(keys.split('.'));
    }

    /// Access a key path in `self` and attempt to convert its type to `T`,
    /// returning `Ok(T)` if the complete path exists and the type is
    /// convertible, `Err` otherwise.
    pub fn get_path_ok_into<'a, 'de, K, T>(&self, keys: K) -> ConfigResult<T>
    where
        T: Deserialize<'de>,
        K: IntoIterator<Item = &'a str>,
    {
        return self.get_path_ok(keys)
            .and_then(|x| {
                x.clone()
                    .try_into()
                    .map_err(|_| {
                        ConfigError::FailedTypeConversion(x.to_string())
                    })
            });
    }

    /// Access a key path in `self`, where individual keys in the path are
    /// separated by `'.'`, and attempt to convert its type to `T`, returning
    /// `Ok(T)` if the complete path exists and the type is convertible, `Err`
    /// otherwise.
    pub fn get_ok_into<'de, T>(&self, keys: &str) -> ConfigResult<T>
    where T: Deserialize<'de>
    {
        return self.get_ok(keys)
            .and_then(|x| {
                x.clone()
                    .try_into()
                    .map_err(|_| {
                        ConfigError::FailedTypeConversion(x.to_string())
                    })
            });
    }

    /// Alias for [`Self::get_ok_into`].
    pub fn a<'de, T>(&self, keys: &str) -> ConfigResult<T>
    where T: Deserialize<'de>
    {
        return self.get_ok_into(keys);
    }

    /// Convert `self` to a bare [`Table`].
    pub fn into_table(self) -> Table { self.data }
}

impl From<ConfigUnver> for Table {
    fn from(config: ConfigUnver) -> Self { config.data }
}

pub extern crate toml;

/// Define a simple function, `load_config` to read and process independent
/// values from a table in a `.toml` file and return them as a `Config` struct.
///
/// Each value to be read is specified as a key (`&'static str`), a default
/// value and its type, the name and type of the `Config` struct field it's
/// stored under, and a closure to map the value read from the file (or the
/// default) to the one stored in the struct. Can only be used once per
/// namespace.
#[macro_export]
macro_rules! config_fn {
    (
        $subtab:expr => {
            $( $key:expr, $def:expr, $intype:ty
                => $field:ident : $outtype:ty = $foo:expr ),+ $(,)?
        }
    ) => {
        #[derive(Clone, Debug)]
        pub struct Config {
            $(
                pub $field: $outtype,
            )+
        }

        pub fn load_config(infile: std::path::PathBuf) -> Config {
            let infile: std::path::PathBuf
                = infile.unwrap_or(std::path::PathBuf::from($file));
            let table: $crate::config::toml::Value
                = std::fs::read_to_string(infile.clone())
                .expect(
                    format!("Couldn't read config file {:?}", infile)
                    .as_str()
                )
                .parse::<$crate::config::toml::Value>()
                .expect(
                    format!("Couldn't parse config file {:?}", infile)
                    .as_str()
                );
            let mut config = Config {
                $(
                    $field: ($foo)($def),
                )+
            };
            if let Some(value) = table.get($subtab) {
                $(
                    if let Some(X) = value.get($key) {
                        let x: $intype
                            = X.clone().try_into()
                            .expect(format!(
                                "Couldn't coerce type for key {:?}", $key)
                                .as_str()
                            );
                        config.$field = ($foo)(x);
                    }
                )+
            }
            return config;
        }
    }
}

