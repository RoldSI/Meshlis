use std::sync::{Arc, Mutex, RwLock, Weak};

use egui_plot::{Legend, Line, Plot, PlotBounds, PlotPoints};
use tokio::net::{TcpListener, TcpStream};
use tokio::io::AsyncReadExt;
use messaging::AudioMessage;
use ringbuffer::ConstGenericRingBuffer;

use eframe::egui;

const BUFFER_SIZE: usize = 2048 << 1;
type Data = Arc<RwLock<Vec<Weak<Mutex<ringbuffer::ConstGenericRingBuffer<f32, BUFFER_SIZE>>>>>>;

struct App {
    data: Data,
}

impl App {
    fn new(data: Data, _cc: &eframe::CreationContext<'_>) -> Self {
        Self {
            data,
        }
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {

        for (i, d) in self.data.read().unwrap().iter().enumerate() {
            if let Some(d) = d.upgrade() {
                egui::Window::new(format!("client-{i}")).show(ctx, |ui| {
                    Plot::new("data")
                        .legend(Legend::default())
                        .allow_zoom(false)
                        .allow_drag(false)
                        .auto_bounds(egui::Vec2b::new(true, false))
                    .show(ui, |ui| {
                        let data = d.lock().unwrap().clone().into_iter().enumerate()
                            .map(|(a, b)| [a as f64, (b as f64) * 5.0]);
                        let data = PlotPoints::from_iter(data);
                        ui.line(Line::new(data));
                    })
                });
            }
        }

        ctx.request_repaint();
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let data = Arc::new(RwLock::new(Vec::new()));

    tokio::spawn({
        let data = Arc::clone(&data);
        async move {
            let listener = TcpListener::bind(("0.0.0.0", 3000)).await?;
            while let Ok((stream, sock)) = listener.accept().await {
                println!("{sock} connected");

                tokio::spawn(listen(stream, Arc::clone(&data)));
            }

            Ok::<_, anyhow::Error>(())
        }
    });

    let native_options = eframe::NativeOptions::default();
    eframe::run_native(
        "server",
        native_options, Box::new(move |cc|
            Box::new(App::new(data, cc))
        ),
    ).unwrap();

    Ok(())
}

async fn listen(
    mut stream: TcpStream,
    d: Data,
) -> anyhow::Result<()> {
    let v = Arc::new(Mutex::new(ConstGenericRingBuffer::<f32, BUFFER_SIZE>::new()));
    d.write().unwrap().push(Arc::downgrade(&v));

    while let Ok(len) = stream.read_u32().await {
        let mut data = vec![0u8; len as _];
        stream.read_exact(&mut data).await?;
        let data: AudioMessage = postcard::from_bytes(&data[..])?;
        v.lock().unwrap().extend(data.data.iter().copied());
    }

    drop(v);
    d.write().unwrap().retain(|v| v.upgrade().is_some());
    Ok(())
}

