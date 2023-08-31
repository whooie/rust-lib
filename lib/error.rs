//! Provides `ErrMsg`, a simple trait to associate a constant string with an
//! error type, and `mkerr`, a macro to easily implement it and error traits.

/// Simple trait to associate a constant string with an error type.
pub trait ErrMsg {
    fn msg(&self) -> &'static str;
}

/// Simple macro to implement `ErrMsg`, `Display`, and `Error` for an error
/// type.
#[macro_export]
macro_rules! mkerr {
    ( $name:ident : { $( $var:ident => $msg:literal ),+ $(,)? } ) => {
        #[derive(Clone, Copy, Debug)]
        pub enum $name {
            $( $var, )+
        }

        impl $crate::error::ErrMsg for $name {
            fn msg(&self) -> &'static str {
                return match *self {
                    $( $name::$var => $msg, )+
                }
            }
        }

        impl std::fmt::Display for $name {
            fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                return f.write_str($crate::error::ErrMsg::msg(self));
            }
        }

        impl std::error::Error for $name { }
    }
}

