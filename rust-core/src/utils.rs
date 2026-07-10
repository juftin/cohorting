/// Reciprocal of 2^64; multiply instead of divide to produce float in [0, 1).
pub const INV_2_64: f64 = 1.0 / ((1u128 << 64) as f64);
