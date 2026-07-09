use pyo3::prelude::*;
use blake2::{Blake2b512, Digest};
use twox_hash::xxh3;

use crate::utils::INV_2_64;

/// Hash a single identifier to a float in [0, 1).
///
/// sep_salt is the pre-encoded b"\x00" + salt bytes from Python.
#[inline]
pub fn hash_single_inner(id_bytes: &[u8], sep_salt: &[u8], use_xxhash: bool) -> f64 {
    if use_xxhash {
        let mut buf = Vec::with_capacity(id_bytes.len() + sep_salt.len());
        buf.extend_from_slice(id_bytes);
        buf.extend_from_slice(sep_salt);
        xxh3::hash64(&buf) as f64 * INV_2_64
    } else {
        let mut hasher = Blake2b512::new();
        hasher.update(id_bytes);
        hasher.update(sep_salt);
        let result = hasher.finalize();
        u64::from_le_bytes(result[..8].try_into().unwrap()) as f64 * INV_2_64
    }
}

/// Hash a single string identifier to a float in [0, 1).
#[pyfunction]
pub fn hash_single(id: String, sep_salt: Vec<u8>, use_xxhash: bool) -> f64 {
    hash_single_inner(id.as_bytes(), &sep_salt, use_xxhash)
}

/// Hash a list of string identifiers to floats in [0, 1).
#[pyfunction]
pub fn hash_strings(
    ids: Vec<String>,
    sep_salt: Vec<u8>,
    use_xxhash: bool,
) -> Vec<f64> {
    ids.iter()
        .map(|id| hash_single_inner(id.as_bytes(), &sep_salt, use_xxhash))
        .collect()
}

/// Return a single random float in [0, 1) sourced from OS entropy.
#[pyfunction]
pub fn random_float() -> f64 {
    let mut buf = [0u8; 8];
    getrandom::getrandom(&mut buf).unwrap();
    u64::from_le_bytes(buf) as f64 * INV_2_64
}

/// Return n random floats in [0, 1) sourced from OS entropy.
#[pyfunction]
pub fn random_floats(n: usize) -> Vec<f64> {
    let mut buf = vec![0u8; n * 8];
    getrandom::getrandom(&mut buf).unwrap();
    buf.chunks_exact(8)
        .map(|chunk| u64::from_le_bytes(chunk.try_into().unwrap()) as f64 * INV_2_64)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash_single_deterministic_blake2b() {
        let a = hash_single_inner(b"user_1", b"\x00exp", false);
        let b = hash_single_inner(b"user_1", b"\x00exp", false);
        assert_eq!(a, b);
        assert!((0.0..1.0).contains(&a));
    }

    #[test]
    fn test_hash_single_deterministic_xxhash() {
        let a = hash_single_inner(b"user_1", b"\x00exp", true);
        let b = hash_single_inner(b"user_1", b"\x00exp", true);
        assert_eq!(a, b);
        assert!((0.0..1.0).contains(&a));
    }

    #[test]
    fn test_hash_different_salts() {
        let a = hash_single_inner(b"user_1", b"\x00exp1", false);
        let b = hash_single_inner(b"user_1", b"\x00exp2", false);
        assert_ne!(a, b);
    }

    #[test]
    fn test_hash_different_ids() {
        let a = hash_single_inner(b"user_1", b"\x00exp", false);
        let b = hash_single_inner(b"user_2", b"\x00exp", false);
        assert_ne!(a, b);
    }

    #[test]
    fn test_hash_strings_matches_single() {
        let sep_salt = b"\x00exp".to_vec();
        let ids: Vec<String> = vec!["a".into(), "b".into(), "c".into()];
        let batch = hash_strings(ids.clone(), sep_salt.clone(), false);
        for (i, id) in ids.iter().enumerate() {
            assert_eq!(batch[i], hash_single(id.clone(), sep_salt.clone(), false));
        }
    }

    #[test]
    fn test_random_float_in_range() {
        let r = random_float();
        assert!((0.0..1.0).contains(&r));
    }

    #[test]
    fn test_random_floats_count() {
        let r = random_floats(10);
        assert_eq!(r.len(), 10);
        for v in &r {
            assert!((0.0..1.0).contains(v));
        }
    }
}
