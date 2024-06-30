use std::net::SocketAddr;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex, RwLock, Weak};

use egui_plot::{Legend, Line, Plot, PlotPoints};
use tokio::net::{TcpListener, TcpStream};
use tokio::io::AsyncReadExt;
use messaging::AudioMessage;
use ringbuffer::{ConstGenericRingBuffer, RingBuffer};

use eframe::egui;
use easyfft::prelude::*;

const BUFFER_SIZE: usize = 2048 << 1;
type InnerData = RwLock<Vec<Weak<(SocketAddr, Mutex<ringbuffer::ConstGenericRingBuffer<f32, BUFFER_SIZE>>)>>>;
type Data = Arc<InnerData>;

struct App { data: Data, should_close: Arc<AtomicBool>, }
impl App {
    fn new(data: Data, should_close: Arc<AtomicBool>, _cc: &eframe::CreationContext<'_>) -> Self {
        Self { data, should_close }
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        if self.should_close.load(std::sync::atomic::Ordering::Relaxed) {
            std::process::exit(0);
        }


        for d in self.data.read().unwrap().iter() {
            if let Some(d) = d.upgrade() {

                let data = d.1.lock().unwrap();

                let fft_data: Vec<_> = data.iter().copied().collect();
                let fft_data = fft_data.real_fft();
                let fft_data = fft_data
                    .into_iter().enumerate()
                    .map(|(a, b)| [a as f64, b.norm() as f64]);

                let data = data.clone().into_iter().enumerate()
                    .map(|(a, b)| [a as f64, (b as f64) * 5.0]);
                egui::Window::new(format!("client-{}", d.0)).show(ctx, |ui| {
                    Plot::new("data")
                        .legend(Legend::default())
                        .allow_zoom(false)
                        .allow_drag(false)
                        .auto_bounds(egui::Vec2b::new(true, false))
                    .show(ui, |ui| {
                        let data = PlotPoints::from_iter(data);
                        ui.line(Line::new(data));
                        let data = PlotPoints::from_iter(fft_data);
                        ui.line(Line::new(data).color(egui::Color32::GREEN));
                    })
                });
            }
        }

        ctx.request_repaint();
    }
}

async fn listen(
    sock: SocketAddr,
    mut stream: TcpStream,
    d: Data,
) -> anyhow::Result<()> {
    let v = Arc::new((sock, Mutex::new(ConstGenericRingBuffer::<f32, BUFFER_SIZE>::new())));
    d.write().unwrap().push(Arc::downgrade(&v));

    while let Ok(len) = stream.read_u32().await {
        let mut data = vec![0u8; len as _];
        stream.read_exact(&mut data).await?;
        let data: AudioMessage = postcard::from_bytes(&data[..])?;
        v.1.lock().unwrap().extend(data.data.iter().copied());
    }

    drop(v);
    d.write().unwrap().retain(|v| v.upgrade().is_some());
    Ok(())
}



pub fn init(port: u16) -> anyhow::Result<Context> {
    let data = Arc::new(RwLock::new(Vec::new()));

    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
    .build()?;

    rt.spawn({
        let data = Arc::clone(&data);
        async move {
            let listener = TcpListener::bind(("0.0.0.0", port)).await?;
            while let Ok((stream, sock)) = listener.accept().await {
                println!("{sock} connected");

                tokio::spawn(listen(sock, stream, Arc::clone(&data)));
            }

            Ok::<_, anyhow::Error>(())
        }
    });

    Ok(Context { data, _rt: rt, should_close: Arc::new(AtomicBool::new(false)) })
}

pub struct Context {
    data: Data,
    should_close: Arc<AtomicBool>,
    _rt: tokio::runtime::Runtime,
}

impl Context {
    pub fn open_window(&self) {
        let data = Arc::clone(&self.data);
        let mut native_options = eframe::NativeOptions::default();
        native_options.run_and_return = true;

        let should_close = Arc::clone(&self.should_close);
        eframe::run_native(
            "server",
            native_options, Box::new(move |cc|
                Box::new(App::new(data, should_close, cc))
            ),
        ).unwrap();
    }

    pub fn close_window(&self) {
        self.should_close.store(true, std::sync::atomic::Ordering::Relaxed);
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        self.close_window();
    }
}

