use pyo3::prelude::*;

mod cohort;
mod hash;
mod utils;

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hash::hash_single, m)?)?;
    m.add_function(wrap_pyfunction!(hash::hash_strings, m)?)?;
    m.add_function(wrap_pyfunction!(hash::hash_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(hash::random_float, m)?)?;
    m.add_function(wrap_pyfunction!(hash::random_floats, m)?)?;
    m.add_function(wrap_pyfunction!(hash::random_floats_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(cohort::assign_single, m)?)?;
    m.add_function(wrap_pyfunction!(cohort::assign_strings, m)?)?;
    m.add_function(wrap_pyfunction!(cohort::assign_numpy, m)?)?;
    Ok(())
}
