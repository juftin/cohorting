use pyo3::prelude::*;

use crate::hash::hash_single_inner;

/// Assign a single string identifier to a cohort.
///
/// lower_bounds must be sorted ascending. cohort_names[i] corresponds to the
/// bucket whose lower bound is lower_bounds[i].
#[pyfunction]
pub fn assign_single(
    id: String,
    sep_salt: Vec<u8>,
    cohort_names: Vec<String>,
    lower_bounds: Vec<f64>,
) -> String {
    let hash_val = hash_single_inner(id.as_bytes(), &sep_salt);
    let idx = lower_bounds.partition_point(|&lb| lb <= hash_val) - 1;
    cohort_names[idx].clone()
}

/// Assign a list of string identifiers to cohorts.
#[pyfunction]
pub fn assign_strings(
    ids: Vec<String>,
    sep_salt: Vec<u8>,
    cohort_names: Vec<String>,
    lower_bounds: Vec<f64>,
) -> Vec<String> {
    ids.iter()
        .map(|id| {
            let hash_val = hash_single_inner(id.as_bytes(), &sep_salt);
            let idx = lower_bounds.partition_point(|&lb| lb <= hash_val) - 1;
            cohort_names[idx].clone()
        })
        .collect()
}

/// Assign cohorts to a numpy array of string identifiers.
///
/// Returns cohort names as strings. The caller wraps back to numpy in Python.
#[pyfunction]
pub fn assign_numpy(
    ids: Bound<'_, PyAny>,
    sep_salt: Vec<u8>,
    cohort_names: Vec<String>,
    lower_bounds: Vec<f64>,
) -> PyResult<Vec<String>> {
    let flat = ids.call_method0("flatten")?;
    let strings: Vec<String> = flat.call_method0("tolist")?.extract()?;
    let result: Vec<String> = strings
        .iter()
        .map(|s| {
            let hash_val = hash_single_inner(s.as_bytes(), &sep_salt);
            let idx = lower_bounds.partition_point(|&lb| lb <= hash_val) - 1;
            cohort_names[idx].clone()
        })
        .collect();
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_assign_single_deterministic() {
        let names = vec!["control".into(), "treatment".into()];
        let lowers = vec![0.0, 0.5];
        let sep_salt = b"\x00exp".to_vec();

        let result1 = assign_single(
            "user_1".into(), sep_salt.clone(), names.clone(), lowers.clone(),
        );
        let result2 = assign_single(
            "user_1".into(), sep_salt.clone(), names.clone(), lowers.clone(),
        );
        assert_eq!(result1, result2);
        assert!(names.contains(&result1));
    }

    #[test]
    fn test_assign_strings_matches_single() {
        let names = vec!["control".into(), "treatment".into()];
        let lowers = vec![0.0, 0.5];
        let sep_salt = b"\x00exp".to_vec();
        let ids: Vec<String> = vec!["a".into(), "b".into(), "c".into()];

        let batch = assign_strings(
            ids.clone(), sep_salt.clone(), names.clone(), lowers.clone(),
        );
        for (i, id) in ids.iter().enumerate() {
            assert_eq!(
                batch[i],
                assign_single(
                    id.clone(), sep_salt.clone(), names.clone(), lowers.clone(),
                )
            );
        }
    }

    #[test]
    fn test_assign_three_way_split() {
        let names = vec!["a".into(), "b".into(), "c".into()];
        let lowers = vec![0.0, 0.3333333333333333, 0.6666666666666666];
        let sep_salt = b"\x00exp".to_vec();

        let result = assign_single("user_1".into(), sep_salt, names.clone(), lowers);
        assert!(names.contains(&result));
    }
}
