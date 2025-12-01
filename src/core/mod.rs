// Ours
// pub mod real;
// pub use real::*;
pub mod scale;
pub mod rep;
pub mod fp4;
pub mod fp6;
pub mod fp8;
pub mod exp2i;

#[cfg(test)]
#[path = "tests/mod.rs"]
mod tests;


// Macros
#[macro_export]
macro_rules! fp8_e5 {
    ($lit:literal) => {
        $crate::core::fp8::Fp8E5M2::from($lit as f32)
    };
}

#[macro_export]
macro_rules! fp8_e4 {
    ($lit:literal) => {
        $crate::core::fp8::Fp8E4M3::from($lit as f32)
    };
}

