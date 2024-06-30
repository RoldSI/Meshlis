
use pyo3::prelude::*;
use pyo3::exceptions;
use server::Context;

#[pyclass]
struct Server(Context);

#[pymethods]
impl Server {
    #[new]
    #[pyo3(signature = (port=3000))]
    fn new(port: u16) -> PyResult<Self> {
        Ok(
            Self(server::init(port)
                .map_err(|err|
                    exceptions::PyConnectionError::new_err(format!("error starting server: {err}"))
                )?)
        )
    }

    fn open_window(self_: PyRef<'_, Self>) { self_.0.open_window() }
    fn close_window(self_: PyRef<'_, Self>) { self_.0.close_window() }
    fn process(_self_: PyRef<'_, Self>) {
        // do processing
    }
}

#[pymodule]
fn pyserver(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Server>()?;
    Ok(())
}

