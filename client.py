import socket
import numpy as np
import matplotlib.pyplot as plt

# Network parameters
HOST = '0.0.0.0'  # Listen on all interfaces
PORT = 5000
CHUNK = 1024
FORMAT = np.int16  # Match the client's audio format
RATE = 44100

# Create a socket connection
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((HOST, PORT))
server_socket.listen(1)
print("Server listening on port", PORT)

conn, addr = server_socket.accept()
print('Connected by', addr)

def perform_fft(data):
    # Convert byte data to numpy array
    np_data = np.frombuffer(data, dtype=FORMAT)
    # Perform FFT
    fft_data = np.fft.fft(np_data)
    # Calculate frequencies
    freqs = np.fft.fftfreq(len(fft_data), 1.0/RATE)
    return freqs, fft_data

try:
    while True:
        data = conn.recv(CHUNK)
        if not data:
            break
        freqs, fft_data = perform_fft(data)
        
        # Plot the FFT result (optional)
        plt.plot(freqs[:len(freqs)//2], np.abs(fft_data)[:len(freqs)//2])
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude')
        plt.title('FFT of Received Audio Data')
        plt.show()
except KeyboardInterrupt:
    print("Stopping server...")

# Close the connection
conn.close()
server_socket.close()