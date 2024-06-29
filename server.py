import socket
import pyaudio

# Audio parameters
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024

# Network parameters
HOST = 'server_ip_address'  # Replace with your server IP address
PORT = 5000

# Initialize PyAudio
audio = pyaudio.PyAudio()

# Open the stream
stream = audio.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

# Create a socket connection
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((HOST, PORT))

print("Streaming audio...")

try:
    while True:
        data = stream.read(CHUNK)
        client_socket.sendall(data)
except KeyboardInterrupt:
    print("Stopping stream...")

# Close the stream and socket
stream.stop_stream()
stream.close()
audio.terminate()
client_socket.close()