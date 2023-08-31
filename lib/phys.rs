//! Collection of physical constants and dimensional units.
//!
//! Values are taken from NIST.

use std::{
    f64::consts::PI,
    ops::Deref,
};

/// Planck constant (kg m^2 s^-1)
pub const h: f64 = 6.62607015e-34;
//             +/- 0 (exact)

/// reduced Planck constant (kg m^2 s^-1)
pub const hbar: f64 = h / 2.0 / PI;
//                +/- 0 (exact)

/// speed of light in vacuum (m s^-1)
pub const c: f64 = 2.99792458e8;
//             +/- 0 (exact)

/// Avogadro's number
pub const NA: f64 = 6.02214076e23;
//              +/- 0 (exact)

/// Boltzmann's constant (J K^-1)
pub const kB: f64 = 1.380649e-23;
//              +/- 0 (exact)

/// electric permittivity in vacuum (F m^-1)
pub const e0: f64 = 8.8541878128e-12;
//              +/- 0.0000000013e-12

/// magnetic permeability in vacuum (N A^-2)
pub const u0: f64 = 1.25663706212e-6;
//              +/- 0.00000000019e-6

/// Newtonian gravitational constant (m^3 kg^-1 s^-2)
pub const G: f64 = 6.67430e-11;
//             +/- 0.00015e-11

/// gravitational acceleration near Earth's surface (m s^-2)
pub const g: f64 = 9.80665;
//             +/- 0 (exact)

/// elementary charge (C)
pub const e: f64 = 1.602176634e-19;
//             +/- 0 (exact)

/// electron mass (kg)
pub const me: f64 = 9.1093837015e-31;
//              +/- 0.0000000028e-31

/// proton mass (kg)
pub const mp: f64 = 1.67262192369e-27;
//              +/- 0.00000000051e-27

/// unified atomic mass unit (kg)
pub const mu: f64 = 1.66053906660e-27;
//              +/- 0.00000000050e-27

/// Rydberg constant for an infinite-mass nucleus (m^-1)
pub const Rinf: f64 = 10973731.568160;
//                       +/- 0.000021

/// fine structure constant
pub const alpha: f64 = 7.2973525693e-3;
//                 +/- 0.0000000011e-3

/// molar gas constant
pub const R: f64 = 8.314462618;
//             +/- 0 (exact)

/// Stefan-Boltzmann constant (W m^-2 K^-4)
pub const SB: f64 = ( PI * PI * kB * kB * kB * kB )
                    / ( 60.0 * hbar * hbar * hbar * c * c );
//              +/- 0 (exact)

/// Bohr radius (m)
pub const a0: f64 = 5.29177210903e-11;
//              +/- 0.00000000080e-11

/// Bohr magneton (J T^-1)
pub const uB: f64 = 9.2740100783e-24;
//              +/- 0.0000000028e-24

/// Hartree energy (J) = 2\*Rinf\*h\*c
pub const Eh: f64 = 4.3597447222071e-18;
//              +/- 0.0000000000085e-18

//  conversion factors
pub const amu2kg: f64 = mu;
pub const kg2amu: f64 = 1.0 / amu2kg;

pub const a02m: f64 = a0;
pub const m2a0: f64 = 1.0 / a0;

pub const cm2J: f64 = 100.0 * h * c;
pub const J2cm: f64 = 1.0 / cm2J;

pub const Hz2J: f64 = h;
pub const J2Hz: f64 = 1.0 / Hz2J;

pub const K2J: f64 = kB;
pub const J2K: f64 = 1.0 / K2J;

pub const Eh2J: f64 = Eh;
pub const J2Eh: f64 = 1.0 / Eh2J;

pub const eV2J: f64 = e;
pub const J2eV: f64 = 1.0 / eV2J;

macro_rules! mkunit {
    (
        $name:ident : { $( $var:ident => $val:expr ),+ $(,)? } ( $def:ident )
    ) => {
        #[derive(Clone, Copy, Debug, PartialEq)]
        pub enum $name {
            Any(f64),
            $( $var, )+
        }

        impl Default for $name {
            fn default() -> Self { $name::$def }
        }

        impl From<f64> for $name {
            fn from(U: f64) -> $name { $name::Any(U) }
        }

        impl From<$name> for f64 {
            fn from(U: $name) -> f64 {
                return match U {
                    $name::Any(u) => u,
                    $( $name::$var => $val ),+
                };
            }
        }

        impl $name {
            pub fn f(self) -> f64 { f64::from(self) }

            pub fn rescale<U>(self, u: U) -> Self
                where U: Into<Self> + Copy
            {
                let x: f64 = u.into().into();
                return $name::Any(self.f() / x);
            }
        }

        impl AsRef<f64> for $name {
            fn as_ref(&self) -> &f64 {
                return match self {
                    &$name::Any(ref f) => f,
                    $( &$name::$var => &$val ),+
                };
            }
        }

        impl Deref for $name {
            type Target = f64;

            fn deref(&self) -> &Self::Target { self.as_ref() }
        }
    }
}

mkunit!(
    LengthUnit : {
        Kilometer => 1.0e3,
        Meter => 1.0,
        Centimeter => 1.0e-2,
        Millimeter => 1.0e-3,
        Micrometer => 1.0e-6,
        Micron => 1.0e-6,
        Nanometer => 1.0e-9,
        Picometer => 1.0e-12,
        Femtometer => 1.0e-15,
        AU => 149597870700.0,
        LightYear => 9460730472580800.0,
        Parsec => 30856775814671900.0,
        Angstrom => 1.0e-10,
        Bohr => a0,
        Planck => 1.616199e-35,
        Compton => 2.426310215e-12,
        Mile => 1609.0,
        Yard => 0.9144,
        Inch => 0.0254,
        Furlong => 201.125
    } ( Meter )
);

mkunit!(
    EnergyUnit : {
        Joule => 1.0,
        Wavenumber => 100.0 * h * c,
        Hertz => h,
        Kelvin => kB,
        Hartree => Eh,
        Electronvolt => e
    } ( Joule )
);

impl EnergyUnit {
    /// Computes an energy scale equal to the ground-state energy of a quantum
    /// particle of mass `m` confined to a box of length `L`, up to a factor of
    /// (2 pi)^2.
    pub fn from_confinement(m: f64, L: f64) -> Self
    {
        return EnergyUnit::Any(hbar.powi(2) / (2.0 * m * L.powi(2)));
    }
}

mkunit!(
    TimeUnit : {
        Millennium => 31557600000.0,
        Century => 3155760000.0,
        Decade => 315576000.0,
        Year => 31557600.0,
        Fortnight => 1209600.0,
        Week => 604800.0,
        Day => 86400.0,
        Hour => 3600.0,
        Minute => 60.0,
        Second => 1.0,
        Millisecond => 1.0e-3,
        Microsecond => 1.0e-6,
        Nanosecond => 1.0e-9,
        Picosecond => 1.0e-12,
        Femtosecond => 1.0e-15,
        Jiffy => 1.0e-15 / c,
        Planck => 5.391059570951581e-44,
    } ( Second )
);


