
use std::{net::SocketAddr, sync::Arc};

use cpal::{traits::{DeviceTrait, HostTrait, StreamTrait}, InputCallbackInfo};
use messaging::AudioMessage;
use tokio::{net::TcpStream, sync::mpsc};
use tokio::io::AsyncWriteExt;
use clap::Parser;

#[derive(Debug, clap::Parser)]
struct App {
    /// the address to connect to (localhost if undefined)
    #[arg(short, long, default_value_t = SocketAddr::from(([127, 0, 0, 1], 3000)))]
    addr: SocketAddr,
}


#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = App::parse();
    let host = cpal::default_host();
    let device = host.default_input_device().expect("no input device");
    let mut supported_configs_range = device.supported_input_configs()
        .expect("no supported input config");
    let config = supported_configs_range.next()
        .expect("no supported config")
        .with_max_sample_rate().config();

    let (tx, rx) = mpsc::channel::<Arc<[f32]>>(16);

    let stream = device.build_input_stream(
        &config,
        move |data: &[f32], _: &InputCallbackInfo| {
            tx.blocking_send(Vec::from(data).into()).unwrap();
        },
        move |err| panic!("error: {err}"),
        None,
    )?;

    stream.play()?;

    print!("connecting... ");
    let stream = TcpStream::connect(args.addr).await?;
    println!("connected");
    sender(stream, rx).await?;
    Ok(())
}


async fn sender(
    mut s: TcpStream, 
    mut rx: mpsc::Receiver<Arc<[f32]>>,
) -> anyhow::Result<()> {
    while let Some(v) = rx.recv().await {
        let data = AudioMessage::new(v);
        let data = postcard::to_allocvec(&data)?;
        s.write_u32(data.len() as _).await?;
        println!("sending");
        s.write_all(&*data).await?;
    }

    println!("disconnected");
    Ok(())
}

