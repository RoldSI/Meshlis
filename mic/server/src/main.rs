
fn main() -> anyhow::Result<()> {
    let v = server::init(3000)?;
    v.open_window();

    Ok(())
}

// optimize phase shift

