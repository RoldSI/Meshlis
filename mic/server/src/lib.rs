use std::net::SocketAddr;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex, RwLock, Weak};

use egui_plot::{Legend, Line, Plot, PlotPoints, Points};
use tokio::net::{TcpListener, TcpStream};
use tokio::io::AsyncReadExt;
use messaging::AudioMessage;
use ringbuffer::{ConstGenericRingBuffer, RingBuffer};

use eframe::egui;
use easyfft::prelude::*;

const BUFFER_SIZE: usize = 2048 << 1;
type InnerData = RwLock<Vec<Weak<(SocketAddr, Mutex<ringbuffer::ConstGenericRingBuffer<f32, BUFFER_SIZE>>)>>>;
type Data = Arc<InnerData>;

struct App {
    data: Data,
    pos: Arc<Mutex<(f32, f32)>>,
    should_close: Arc<AtomicBool>,
    pos_fn: Arc<Mutex<Option<Box<dyn FnMut(Vec<Vec<f32>>) -> (f32, f32)>>>>,
    ctx: Arc<Context>,
}
impl App {
    fn new(
        data: Data,
        should_close: Arc<AtomicBool>,
        pos: Arc<Mutex<(f32, f32)>>,
        pos_fn: Arc<Mutex<Option<Box<dyn FnMut(Vec<Vec<f32>>) -> (f32, f32)>>>>,
        ctx: Arc<Context>,
        _cc: &eframe::CreationContext<'_>,
    ) -> Self {
        Self { data, should_close, pos_fn, pos, ctx }
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        if self.should_close.load(std::sync::atomic::Ordering::Relaxed) {
            std::process::exit(0);
        }

        self.ctx.call_pos();

        egui::Window::new("position").show(ctx, |ui| {
            Plot::new("pos")
                .legend(Legend::default())
                .auto_bounds(egui::Vec2b::new(false, false))
            .show(ui, |ui| {
                let (px, py) = *self.pos.lock().unwrap();
                ui.points(Points::new(vec![
                    [px as _, py as _]
                ]).radius(10.0))
            });
        });

        for d in self.data.read().unwrap().iter() {
            if let Some(d) = d.upgrade() {

                let data = d.1.lock().unwrap();

                let fft_data: Vec<_> = data.iter().copied().collect();
                let fft_data = fft_data.real_fft();
                // 3000 hz
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
    let pos = Arc::new(Mutex::new((0.0, 0.0)));
    let pos_fn = Arc::new(Mutex::new(None));

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

    Ok(Context {
        data, _rt: Arc::new(rt),
        should_close: Arc::new(AtomicBool::new(false)),
        pos_fn,
        pos,
    })
}

#[derive(Clone)]
pub struct Context {
    data: Data,
    pos: Arc<Mutex<(f32, f32)>>,
    should_close: Arc<AtomicBool>,
    pos_fn: Arc<Mutex<Option<Box<dyn FnMut(Vec<Vec<f32>>) -> (f32, f32)>>>>,
    _rt: Arc<tokio::runtime::Runtime>,
}

impl Context {
    pub fn open_window(self: Arc<Self>) {
        let data = Arc::clone(&self.data);
        let mut native_options = eframe::NativeOptions::default();
        native_options.run_and_return = true;

        let should_close = Arc::clone(&self.should_close);
        let pos = Arc::clone(&self.pos);
        let pos_fn = Arc::clone(&self.pos_fn);
        eframe::run_native(
            "server",
            native_options, Box::new(move |cc|
                Box::new(App::new(data, should_close, pos, pos_fn, Arc::clone(&self), cc))
            ),
        ).unwrap();
    }

    pub fn close_window(&self) {
        self.should_close.store(true, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn data(&self) -> Vec<Vec<f32>> {
        self.data.read().unwrap().iter()
            .filter_map(|v| v.upgrade().map(|v| {
                v.1.lock().unwrap().to_vec()
            }))
        .collect()
    }

    pub fn store_pos(&self, pos: (f32, f32)) { *self.pos.lock().unwrap() = pos }
    pub fn register_pos(&self, v: impl FnMut(Vec<Vec<f32>>) -> (f32, f32)) {
        *self.pos_fn.lock().unwrap() = Some(unsafe {
            let b: Box<dyn FnMut(Vec<Vec<f32>>) -> (f32, f32)> = Box::new(v);
            std::mem::transmute(b)
        });
    }
    pub fn call_pos(&self) {
        let mut v = self.pos_fn.lock().unwrap();
        if let Some(v) = &mut *v {
            println!("calling");
            self.store_pos(v(self.data().iter().map(|v| {
                v.real_fft().to_vec().into_iter().map(|v| v.re).skip(1400).take(1550).collect()
            }).collect()));
        }
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        self.close_window();
    }
}

