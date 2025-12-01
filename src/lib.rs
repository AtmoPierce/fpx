#![cfg_attr(feature = "f16", feature(f16))]
#![cfg_attr(feature = "f128", feature(f128))]

pub mod core;
pub use core::*;

#[cfg(feature = "design")]
pub mod design;
#[cfg(feature = "design")]
pub use design::*;