import socket
import struct
import time

import pyaudio

# Audio parameters
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
global stream
# Network parameters
HOST = 'server_ip_address'  # Replace with your server IP address
PORT = 5000
audio = pyaudio.PyAudio()  # init pyaudio


def send_audio_packet() -> None:
    data = stream.read(CHUNK)
    timestamp = time.time()
    client_socket.sendall(struct.pack('d', timestamp) + data)


if __name__ == "__main__":
    print("Opening audio stream")
    stream = audio.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)
    print("Connecting socket")
    client_socket.connect((HOST, PORT))
    print("Streaming audio...")
    try:
        while True:
            send_audio_packet()
    except KeyboardInterrupt:
        print("Stopping stream...")
    print("Closing audio stream")
    # Close the stream and socket
    stream.stop_stream()
    stream.close()
    audio.terminate()
    client_socket.close()
