import socket
import threading
import queue

import cv2
import numpy as np

import json

from markerDetector import getCorners

# Create a queue to hold messages that the sender thread needs to send
send_queue = queue.Queue()

CAMERA_RESOLUTION = (1280, 960)
BYTES_PER_PIXEL = 4

def receiver_thread(client_socket, addr):
    """Continuously receives data from the client."""
    print(f"Receiver started for {addr}")
    while True:
        try:
            print("Waiting to receive length...")

            length = b''
            while len(length) < 4:
                remaining = client_socket.recv(4 - len(length))

                if not remaining:
                    print("connection closed by client during length reception")
                    break

                length += remaining

            print(f"Received length bytes from {str(addr)}: {length}")

            image_size = int.from_bytes(length, byteorder='big')

            image = b''
            while len(image) < image_size:
                chunk = client_socket.recv(image_size - len(image))

                if not chunk:
                    print("connection closed by client during image reception")
                    break

                image += chunk

            print(f"Received from {str(addr)} image size {image_size} bytes")

            # Display image
            try:
                # Reshape the raw byte array into a NumPy array
                np_arr = np.frombuffer(image, np.uint8)
                
                # Reshape the 1D array into the image dimensions (H, W, Channels)
                # Note: If client uses Color32 (RGBA), channels=4. 
                # If client uses RGB24, channels=3.
                img = np_arr.reshape((CAMERA_RESOLUTION[1], CAMERA_RESOLUTION[0], BYTES_PER_PIXEL))

                # Convert the color space if necessary (e.g., from RGBA to BGR for OpenCV display)
                if BYTES_PER_PIXEL == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
                elif BYTES_PER_PIXEL == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

                img = cv2.flip(img, 0)  # vertical flip

                ids, corners = getCorners(img)

                print(f"Detected corners: {corners}")
                # Sample 2 corners detected:
                '''
                (
                    [
                        [
                            [604., 603.],
                            [716., 615.],
                            [719., 710.],
                            [610., 697.]
                        ]
                    ],
                    [
                        [
                            [739., 534.],
                            [719., 586.],
                            [610., 575.],
                            [639., 527.]
                        ]
                    ]
                )
                '''

                corners_list = None

                if len(ids) > 0:
                    corners_list = {
                        "id": ids[0],

                        "corner1": corners[0].tolist()[0][0],
                        "corner2": corners[0].tolist()[0][1],
                        "corner3": corners[0].tolist()[0][2],
                        "corner4": corners[0].tolist()[0][3]
                    }

                print("JSON: ", json.dumps(corners_list))
                
                send_queue.put(json.dumps(corners_list).encode('utf-8'))
                
            except Exception as e:
                print(f"Error decoding raw image data: {e}")
            
            # --- Business Logic: Enqueue a response if needed ---
            # Example: Echo the message back or send a canned response
            # response = f"Echo: {message}"
            # send_queue.put(response.encode('ascii'))
            
        except ConnectionResetError:
            break
        except Exception as e:
            print(f"Receiver error: {e}")
            break
    
    print(f"Receiver for {addr} closing.")
    # In a real app, you'd signal the sender thread to stop here

def sender_thread(client_socket, addr):
    """Continuously sends data waiting in the queue to the client."""
    print(f"Sender started for {addr}")
    while True:
        try:
            # Blocks until an item is available in the queue
            data_to_send = send_queue.get() 
            client_socket.sendall(data_to_send)
            send_queue.task_done()
            
        except BrokenPipeError:
            break
        except Exception as e:
            print(f"Sender error: {e}")
            break

def handle_client(client_socket, addr):
    print(f'Got a connection from {str(addr)}')

    # Send an initial welcome message
    # client_socket.send(b'Server says connected') # interferes with corner processing in Unity

    # 1. Start the Receiver thread
    rx_thread = threading.Thread(target=receiver_thread, args=(client_socket, addr))
    rx_thread.start()

    # 2. Start the Sender thread
    tx_thread = threading.Thread(target=sender_thread, args=(client_socket, addr))
    tx_thread.start()

    # Wait for both threads to finish (which happens only when an error occurs)
    rx_thread.join()
    tx_thread.join()

    client_socket.close()
    print(f"Connection with {addr} closed.")

# Create a socket object
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.settimeout(60)

# Get local machine name
host = "127.0.0.1"
port = 65432

# Bind the socket to a public host and port
server_socket.bind((host, port))

# Listen for incoming connections
server_socket.listen(5)
print('Server is listening on port %s...' % port)

while True:
    # Accept a connection
    client_socket, addr = server_socket.accept()

    # Create a new thread to handle the client
    client_thread = threading.Thread(target=handle_client, args=(client_socket, addr))
    client_thread.start()