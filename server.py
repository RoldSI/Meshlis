import socket
import struct
import threading
import time
from collections import defaultdict, deque
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

CHUNK = 1024

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind(('0.0.0.0', 5000))
server_socket.listen(5)

buffers = defaultdict(deque)  # Buffer for each client
client_threads = []
client_colors = {}  # Dictionary to store colors for each client

# Generate a list of colors for clients
colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']

def handle_client(client_socket, client_id):
    while True:
        data = client_socket.recv(CHUNK + 8)  # 8 bytes for the timestamp
        if not data:
            break
        timestamp, chunk = struct.unpack('d', data[:8])[0], data[8:]
        buffers[client_id].append((timestamp, chunk))
    client_socket.close()

def synchronize_audio():
    while True:
        try:
            if buffers:  # Check if there are any clients connected
                # Ensure all buffers have data
                if all(len(buffers[client_id]) > 0 for client_id in buffers):
                    # Fetch the earliest chunk from each client
                    chunks = []
                    for client_id in buffers:
                        if len(buffers[client_id]) > 0:
                            timestamp, chunk = buffers[client_id].popleft()
                            chunks.append((client_id, timestamp, chunk))

                    if chunks:
                        # Find the earliest timestamp
                        min_timestamp = min(timestamp for client_id, timestamp, chunk in chunks)

                        # Collect all chunks with timestamps close to min_timestamp
                        synchronized_chunks = [(client_id, chunk) for client_id, timestamp, chunk in chunks if timestamp - min_timestamp < 0.05]

                        # Process the synchronized chunks (plot the amplitude)
                        plot_audio_amplitude(synchronized_chunks)
                else:
                    time.sleep(0.01)  # Sleep for a short time if buffers are empty
            else:
                time.sleep(0.01)  # Sleep for a short time if no clients are connected
        except Exception as e:
            print(f"Error in synchronize_audio: {e}")

def plot_audio_amplitude(chunks):
    global plot_data

    # Convert chunks to numpy arrays and compute amplitudes
    for client_id, chunk in chunks:
        audio_data = np.frombuffer(chunk, dtype=np.int16)
        amplitude = np.abs(audio_data)

        # Append to the global data for plotting
        if client_id not in plot_data:
            plot_data[client_id] = np.zeros(10000)  # Initialize buffer for new clients

        plot_data[client_id] = np.append(plot_data[client_id], amplitude)
        if len(plot_data[client_id]) > 10000:
            plot_data[client_id] = plot_data[client_id][-10000:]  # Keep only the last 10000 samples

def accept_clients():
    client_count = 0
    while True:
        client_socket, addr = server_socket.accept()
        client_id = addr[1]  # Use port number as client_id for simplicity
        client_thread = threading.Thread(target=handle_client, args=(client_socket, client_id))
        client_threads.append(client_thread)
        client_thread.start()

        # Assign a color to the new client
        client_colors[client_id] = colors[client_count % len(colors)]
        client_count += 1

# Initialize plot data
plot_data = {}

# Plotting setup
fig, ax = plt.subplots()
lines = {}

def update_plot(frame):
    for client_id in plot_data:
        if client_id not in lines:
            # Create a new line for each client
            lines[client_id], = ax.plot(plot_data[client_id], color=client_colors[client_id])
        else:
            lines[client_id].set_ydata(plot_data[client_id])
    return lines.values()

ani = animation.FuncAnimation(fig, update_plot, blit=True, interval=50)

# Start the client acceptance thread
accept_thread = threading.Thread(target=accept_clients)
accept_thread.start()

# Start the synchronization thread
sync_thread = threading.Thread(target=synchronize_audio)
sync_thread.start()

plt.show()
