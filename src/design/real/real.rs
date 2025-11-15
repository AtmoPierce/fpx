pub trait Real:
    Copy
    + PartialEq
    + PartialOrd
    + core::ops::Add<Output = Self>
    + core::ops::Sub<Output = Self>
    + core::ops::Mul<Output = Self>
    + core::ops::Div<Output = Self>
    + core::ops::Neg<Output = Self>
{
    const ZERO: Self;
    const ONE: Self;

    fn from_f32(x: f32) -> Self;
    fn from_f64(x: f64) -> Self;
}

impl Real for f32 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    #[inline] fn from_f32(x: f32) -> Self { x }
    #[inline] fn from_f64(x: f64) -> Self { x as f32 }
}

impl Real for f64 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    #[inline] fn from_f32(x: f32) -> Self { x as f64 }
    #[inline] fn from_f64(x: f64) -> Self { x }
}
