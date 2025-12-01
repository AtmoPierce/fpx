pub mod remez;
pub use crate::remez::remez::*;
pub mod remez_opt;
pub mod create;
pub use create::*;

#[cfg(feature = "plots")]
use aether_viz::*;
#[cfg(feature = "plots")]
#[macro_export]
macro_rules! plots { ($($tt:tt)*) => { $($tt)* } }
#[cfg(not(feature = "plots"))]
#[macro_export]
macro_rules! plots { ($($tt:tt)*) => {}; }
