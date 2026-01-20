import sys

from collections import deque
import socket
import threading
import queue

import cv2
import numpy as np

import json
import time

from markerDetector import getCorners

# Create a queue to hold messages that the sender thread needs to send
send_queue = queue.Queue()

# Buffer to hold lastest image, deque to automatically discard old images
latest_image_buffer = deque(maxlen=1)

# Counter for saving images
image_save_counter = 0
image_save_lock = threading.Lock()
MAX_IMAGES_TO_SAVE = 0

# 3D pose estimation parameters
TAG_SIZE = 0.031 # ArUco tag size (m)
obj_points = np.array([
    [-TAG_SIZE/2,  TAG_SIZE/2, 0],
    [ TAG_SIZE/2,  TAG_SIZE/2, 0],
    [ TAG_SIZE/2, -TAG_SIZE/2, 0],
    [-TAG_SIZE/2, -TAG_SIZE/2, 0]
], dtype=np.float32)

camera_matrix = np.eye(3, dtype=np.float32)
dist_coeffs = np.zeros((4, 1))

# CAMERA_RESOLUTION = (800, 600)
# BYTES_PER_PIXEL = 4

# Image receive resolution
RECEIVE_RESOLUTION = (640.0, 640.0)

def scale_camera_matrix(vals):
    # vals = [fx, fy, cx, cy, orig_w, orig_h]
    # We ignore vals[4] and [5] because the Quest reported 800,600 (incorrectly)
    s = RECEIVE_RESOLUTION[0] / vals[4]
    
    # IMPORTANT: Use the SAME scale for Focal Length to keep the 3D math uniform
    fx_scaled = vals[0] * s
    fy_scaled = vals[1] * s # Use sx here too! 
    
    # Principal Point Scaling
    # If Unity is stretching the image to 800x600:
    cx_scaled = vals[2] * s
    cy_scaled = vals[3] * s
    
    print(f"Scale: {s}, Scaled CX: {cx_scaled}, CY: {cy_scaled}")

    return np.array([
        [fx_scaled, 0,         cx_scaled],
        [0,         fy_scaled, cy_scaled],
        [0,         0,         1.0]
    ], dtype=np.float32)

def process_image_thread():
    global camera_matrix
    global image_save_counter

    # Display image
    while True:
        # Check if image is in buffer
        if len(latest_image_buffer) > 0:
            try:
                # Reshape the raw byte array into a NumPy array
                np_arr = np.frombuffer(latest_image_buffer.pop(), np.uint8)
                
                # Reshape the 1D array into the image dimensions (H, W, Channels)
                # Note: If client uses Color32 (RGBA), channels=4. 
                # If client uses RGB24, channels=3.
                img = cv2.imdecode(np_arr, 1)

                if img is None:
                    print("Failed to decode image")
                    continue
                
                # Save first 20 images to JPG files
                with image_save_lock:
                    if image_save_counter < MAX_IMAGES_TO_SAVE:
                        filename = f"image_{image_save_counter:03d}.jpg"
                        cv2.imwrite(filename, img)
                        print(f"Saved image {image_save_counter + 1}/{MAX_IMAGES_TO_SAVE} to {filename}")
                        image_save_counter += 1
                
                # Convert to grayscale
                grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                result = getCorners(grayscale)

                # Make sure OpenCV result is valid
                if not result or len(result) == 0:
                    continue

                ids, corners = result

                # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                # for corner in corners:
                #     cv2.cornerSubPix(grayscale, corner, (5, 5), (-1, -1), criteria)

                # print(f"Detected corners: {corners}")
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

                marker_data = {}
                
                if len(ids) > 0:
                    image_points = corners[0].reshape(4, 2)

                    success, rvec, tvec = cv2.solvePnP(obj_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_IPPE_SQUARE)
                    if success:
                        rvec, tvec = cv2.solvePnPRefineLM(obj_points, image_points, camera_matrix, dist_coeffs, rvec, tvec)
                    
                    marker_data = {
                        "id": int(ids[0][0]),
                        "tvec": tvec.flatten().tolist(),
                        "rvec": rvec.flatten().tolist()
                    }

                    # print("JSON: ", json.dumps(corners_list))
                    # Convert to JSON for sending
                    json_send_message = json.dumps(marker_data) + '\n'
                    # Put message in send queue
                    send_queue.put(json_send_message.encode('utf-8'))
                    
            except Exception as e:
                print(f"Error decoding raw image data: {e}")
    
    # --- Business Logic: Enqueue a response if needed ---
    # Example: Echo the message back or send a canned response
    # response = f"Echo: {message}"
    # send_queue.put(response.encode('ascii'))

def receiver_thread(client_socket, addr):
    """Continuously receives data from the client."""
    print(f"Receiver started for {addr}")
    global camera_matrix
    
    # --- INTRINSICS PHASE ---
    try:
        # Unity sends "fx,fy,cx,cy" as a raw UTF8 string first
        handshake_data = client_socket.recv(1024).decode('utf-8')
        if handshake_data:
            vals = [float(x) for x in handshake_data.split(',')]
            camera_matrix = scale_camera_matrix(vals)
            print(f"camera intrinsics received:\n{camera_matrix}")
            # Send confirmation back so Unity starts streaming images
            send_queue.put("HANDSHAKE_OK\n".encode('utf-8'))
    except Exception as e:
        print(f"Handshake failed: {e}")
        return
    
    # --- IMAGE RECEIVING PHASE ---
    while True:
        try:
            # print("Waiting to receive length...")

            # Receiver length of incoming image
            length = b''
            while len(length) < 4:
                remaining = client_socket.recv(4 - len(length))

                if not remaining:
                    print("connection closed by client during length reception")
                    sys.exit()
                    break

                length += remaining

            # print(f"Received length bytes from {str(addr)}: {length}")

            image_size = int.from_bytes(length, byteorder='big')

            # Receive image
            image = b''
            while len(image) < image_size:
                chunk = client_socket.recv(image_size - len(image))

                if not chunk:
                    print("connection closed by client during image reception")
                    break

                image += chunk

            # print(f"Received from {str(addr)} image size {image_size} bytes")

            # If buffer is not empty, replace old data; otherwise, append new data
            if len(latest_image_buffer) > 0:
                latest_image_buffer[0] = image
            else:
                latest_image_buffer.append(image)
            
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

    # 3. Start the Image Processor thread
    proc_thread = threading.Thread(target=process_image_thread, args=())
    proc_thread.start()

    # Wait for both threads to finish (which happens only when an error occurs)
    rx_thread.join()
    tx_thread.join()
    proc_thread.join()

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