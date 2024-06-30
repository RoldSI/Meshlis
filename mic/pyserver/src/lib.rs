
use std::sync::Arc;

use pyo3::prelude::*;
use pyo3::exceptions;
use pyo3::types::{PyFunction, PyTuple};
use server::Context;

#[pyclass]
struct Server(Arc<Context>);

unsafe impl Send for Server {}
unsafe impl Sync for Server {}

#[pymethods]
impl Server {
    #[new]
    #[pyo3(signature = (port=3000))]
    fn new(port: u16) -> PyResult<Self> {
        Ok(
            Self(Arc::new(server::init(port)
                .map_err(|err|
                    exceptions::PyConnectionError::new_err(format!("error starting server: {err}"))
                )?))
        )
    }

    fn open_window(self_: PyRef<'_, Self>) { self_.0.clone().open_window() }
    fn close_window(self_: PyRef<'_, Self>) { self_.0.close_window() }
    fn process<'py>(self_: PyRef<'_, Self>, cb: Bound<'py, PyFunction>) {
        self_.0.register_pos(move |d| {
            let cb = cb.clone();
            let (x, y) = Python::with_gil(move |py| {
                (move || {
                    let it = d.into_iter().map(|v| {
                        numpy::PyArray1::from_vec_bound(py, v)
                    });
                    let arr = PyTuple::new_bound(py, it);

                    let res = cb.call1((arr,))?;
                    let x: f32 = res.get_item("x")?.extract()?;
                    let y: f32 = res.get_item("y")?.extract()?;

                    Ok::<_, PyErr>((x, y))
                })().unwrap()
            });

            (x, y)
        });
    }
}

#[pymodule]
fn pyserver(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Server>()?;
    Ok(())
}

