import struct
import sys
import traceback

from collections import deque
import socket
import threading
import queue

from OpenCV_KF import OpenCV_KF
from sliding_window import SlidingWindow

import cv2
import numpy as np
import scipy.spatial.transform

import json
import time
import csv
from datetime import datetime

from markerDetector import getCorners

use_inference = True

try:
    from classification.graspInference import GraspInference

    predictor = GraspInference()
    predictor.start()

except Exception as e:
    use_inference = False
    print(f"GraspInference initialization error: {e}")

# Program running
running = True

# Create a queue to hold messages that the sender thread needs to send
send_queue = queue.Queue()

# Buffer to hold lastest image, deque to automatically discard old images
latest_image_buffer = deque(maxlen=1)

# Buffer to hold latest timestamp
latest_timestamp = 0
latest_cam_pos = np.zeros(3, dtype=np.float32)
latest_cam_rot = np.zeros(4, dtype=np.float32)

# Buffer to hold latest timestamp
latest_timestamp = 0
previous_timestamp = 0


# Counter for saving images
image_save_counter = 0
image_save_lock = threading.Lock()
MAX_IMAGES_TO_SAVE = 0

# 3D pose estimation parameters
TAG_AREA = 0.031 # ArUco tag area (m)
TAG_WIDTH = 0.01323
MARGIN = 0.0045

def get_rotation_matrix_90(axis, num_rotations):
    sin90 = 1
    rot_mat = np.eye(3, dtype=int)
    final_rot = np.eye(3, dtype=int)

    if num_rotations == 0:
        return final_rot
    elif num_rotations > 0:
        sin90 = 1
    else:
        sin90 = -1
    
    if axis == 'x':
        rot_mat = np.array([
            [1, 0, 0],
            [0, 0, -sin90],
            [0, sin90, 0]
        ])
        
    elif axis == 'y':
        rot_mat = np.array([
            [0, 0, sin90],
            [0, 1, 0],
            [-sin90, 0, 0]
        ])
    
    else:
        rot_mat = np.array([
            [0, -sin90, 0],
            [sin90, 0, 0],
            [0, 0, 1]
        ])
    
    for i in range(abs(num_rotations)):
        final_rot @= rot_mat

    return final_rot

template_tag = {
    # Front view
    # TL
    0 :
    np.array([
        [-TAG_AREA / 2, TAG_AREA / 2, 0],
        [-MARGIN / 2, TAG_AREA / 2, 0],
        [-MARGIN / 2, MARGIN / 2, 0],
        [-TAG_AREA / 2, MARGIN / 2, 0]
    ]),

    # TR
    1 :
    np.array([
        [MARGIN / 2, TAG_AREA / 2, 0],
        [TAG_AREA / 2, TAG_AREA / 2, 0],
        [TAG_AREA / 2, MARGIN / 2, 0],
        [MARGIN / 2, MARGIN / 2, 0]
    ]),

    # BL
    2 :
    np.array([
        [-TAG_AREA / 2, -MARGIN / 2, 0],
        [-MARGIN / 2, -MARGIN / 2, 0],
        [-MARGIN / 2, -TAG_AREA / 2, 0],
        [-TAG_AREA / 2, -TAG_AREA / 2, 0]
    ]),

    # BR
    3 :
    np.array([
        [MARGIN / 2, -MARGIN / 2, 0],
        [TAG_AREA / 2, -MARGIN / 2, 0],
        [TAG_AREA / 2, -TAG_AREA / 2, 0],
        [MARGIN / 2, -TAG_AREA / 2, 0]
    ])
}

tag_points_3D = {}
offset = []
rot_axis = 'x'
rot_amount = 0

for i in range(0, 24, 4):
    if i == 0:
        # Top Face
        rot_axis = 'x'
        rot_amount = -1 # Num 90 deg rotations

        offset = np.array([ # Face offset from cube center
            [0, MARGIN + TAG_AREA / 2, 0],
            [0, MARGIN + TAG_AREA / 2, 0],
            [0, MARGIN + TAG_AREA / 2, 0],
            [0, MARGIN + TAG_AREA / 2, 0]
        ])

    elif i == 4:
        # Back face
        rot_axis = 'y'
        rot_amount = -2

        offset = np.array([
            [0, 0, -(MARGIN + TAG_AREA / 2)],
            [0, 0, -(MARGIN + TAG_AREA / 2)],
            [0, 0, -(MARGIN + TAG_AREA / 2)],
            [0, 0, -(MARGIN + TAG_AREA / 2)]
        ])

    elif i == 8:
        # Left Face
        rot_axis = 'y'
        rot_amount = -1

        offset = np.array([
            [-(MARGIN + TAG_AREA / 2), 0, 0],
            [-(MARGIN + TAG_AREA / 2), 0, 0],
            [-(MARGIN + TAG_AREA / 2), 0, 0],
            [-(MARGIN + TAG_AREA / 2), 0, 0]
        ])

    elif i == 12:
        # Front Face
        rot_axis = 'y'
        rot_amount = 0

        offset = np.array([
            [0, 0, MARGIN + TAG_AREA / 2],
            [0, 0, MARGIN + TAG_AREA / 2],
            [0, 0, MARGIN + TAG_AREA / 2],
            [0, 0, MARGIN + TAG_AREA / 2]
        ])

    elif i == 16:
        # Right Face
        rot_axis = 'y'
        rot_amount = 1

        offset = np.array([
            [MARGIN + TAG_AREA / 2, 0, 0],
            [MARGIN + TAG_AREA / 2, 0, 0],
            [MARGIN + TAG_AREA / 2, 0, 0],
            [MARGIN + TAG_AREA / 2, 0, 0]
        ])

    elif i == 20:
        # Bottom Face
        rot_axis = 'x'
        rot_amount = 1

        offset = np.array([
            [0, -(MARGIN + TAG_AREA / 2), 0],
            [0, -(MARGIN + TAG_AREA / 2), 0],
            [0, -(MARGIN + TAG_AREA / 2), 0],
            [0, -(MARGIN + TAG_AREA / 2), 0]
        ])

    tag_points_3D[i] = template_tag[i % 4] @ get_rotation_matrix_90(rot_axis, rot_amount).T + offset
    tag_points_3D[i + 1] = template_tag[(i + 1) % 4] @ get_rotation_matrix_90(rot_axis, rot_amount).T + offset
    tag_points_3D[i + 2] = template_tag[(i + 2) % 4] @ get_rotation_matrix_90(rot_axis, rot_amount).T + offset
    tag_points_3D[i + 3] = template_tag[(i + 3) % 4] @ get_rotation_matrix_90(rot_axis, rot_amount).T + offset

camera_matrix = np.eye(3, dtype=np.float32)
dist_coeffs = np.zeros((4, 1))

# CAMERA_RESOLUTION = (800, 600)
# BYTES_PER_PIXEL = 4

# Image receive resolution
# RECEIVE_RESOLUTION = (640.0, 640.0)
INTRINSICS_RESOLUTION = (1280.0, 1280.0)

MANUAL_FOCAL_LENGTH_ADJUST = 1.00

def scale_camera_matrix(vals):
    # vals = [fx, fy, cx, cy, orig_w, orig_h]
    print(f"Intrinsics: fx={vals[0]}, fy={vals[1]}, cx={vals[2]}, cy={vals[3]}, orig_w={vals[4]}, orig_h={vals[5]}")
    print(INTRINSICS_RESOLUTION[0], INTRINSICS_RESOLUTION[1])
    
    sx = vals[4] / INTRINSICS_RESOLUTION[0]
    sy = vals[5] / INTRINSICS_RESOLUTION[1]
    
    # IMPORTANT: Use the SAME scale for Focal Length to keep the 3D math uniform
    fx_scaled = vals[0] * sx
    fy_scaled = vals[1] * sy
    
    # Principal Point Scaling
    # If Unity is stretching the image to 800x600:
    cx_scaled = vals[2] * sx
    cy_scaled = vals[3] * sy
    
    print(f"Scale x: {sx}, Scale y: {sy}, Scaled CX: {cx_scaled}, CY: {cy_scaled}")

    return np.array([
        [fx_scaled, 0,         cx_scaled],
        [0,         fy_scaled, cy_scaled],
        [0,         0,         1.0]
    ], dtype=np.float32)

    # Manual calibration matrix (from OpenCV calibration tool)
    # return np.array([[430.90683539 * MANUAL_FOCAL_LENGTH_ADJUST,   0,         319.83419697],
    #                  [  0,         431.35829214 * MANUAL_FOCAL_LENGTH_ADJUST, 322.411503  ],
    #                  [  0,            0,            1        ]], dtype=np.float32)

def process_image_thread():
    global camera_matrix
    global image_save_counter
    global latest_timestamp
    global latest_timestamp
    global previous_timestamp

    last_rvec = None
    last_tvec = None
    last_point_count = None
    inliers = None
    
    # Initialize CSV logging
    timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    csv_filename = f"pose_log_{timestamp_str}.csv"
    csv_file = open(csv_filename, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['timestamp', 'raw_rvec_x', 'raw_rvec_y', 'raw_rvec_z', 'raw_tvec_x', 'raw_tvec_y', 'raw_tvec_z', 'filtered_rvec_x', 'filtered_rvec_y', 'filtered_rvec_z', 'filtered_tvec_x', 'filtered_tvec_y', 'filtered_tvec_z', 'pose_success'])
    csv_file.flush()
    print(f"CSV logging initialized: {csv_filename}")
    
    # Initialize Kalman filter
    x = np.zeros((16, 1), dtype=np.double)
    x[9, 0] = 1

    dx_k = np.zeros((15, 1), dtype=np.double)

    x_forward = x.copy()

    sigma_acc =  np.array([1E-1, 1E-1, 1E-1], dtype=np.double)
    sigma_gyro = np.array([8E-2, 8E-2, 8E-2], dtype=np.double)
    Q = np.diag(np.concatenate((sigma_acc**2, sigma_gyro**2)))
    
    pos_var = np.array([8.82E-6, 4.52E-6, 1.47E-6], dtype=np.double)
    rot_var = np.array([8.68E-5, 1.04E-4, 3.47E-4], dtype=np.double)
    R = np.diag(np.concatenate((pos_var, rot_var)))

    P = np.eye(15, dtype=np.double) * 0.5

    kf = OpenCV_KF(x, dx_k, P, Q, R)

    dt = np.array([0], dtype=np.double)

    # Initialize sliding window
    sw_t = SlidingWindow(5, 3)
    sw_r = SlidingWindow(5, 3)

    # Display image
    while running:
        t_0_s = time.perf_counter()

        # Check if image is in buffer
        if len(latest_image_buffer) == 0:
            time.sleep(0.01) # Sleep 10ms to let other threads run
            continue
        t_0_e = time.perf_counter()
        
        try:
            # Reshape the raw byte array into a NumPy array
            # np_arr = np.frombuffer(latest_image_buffer.pop(), np.uint8)
            
            # Reshape the 1D array into the image dimensions (H, W, Channels)
            # Note: If client uses Color32 (RGBA), channels=4. 
            # If client uses RGB24, channels=3.
            # img = cv2.imdecode(np_arr, 1)
            t_1_s = time.perf_counter()
            img = np.frombuffer(latest_image_buffer.pop(), np.uint8).reshape((1280, 1280)) #TODO: Make resolution dynamic based on handshake intrinsics
            img = img = cv2.flip(img, 0)
            
            if img is None:
                print("Failed to decode image")
                continue
            
            t_1_e = time.perf_counter()
            # Save first 20 images to JPG files
            with image_save_lock:
                if image_save_counter < MAX_IMAGES_TO_SAVE:
                    filename = f"image_{image_save_counter:03d}.jpg"
                    cv2.imwrite(filename, img)
                    print(f"Saved image {image_save_counter + 1}/{MAX_IMAGES_TO_SAVE} to {filename}")
                    image_save_counter += 1
            
            # Convert to grayscale
            # grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # result = getCorners(grayscale)
            t_2_s = time.perf_counter()
            result = getCorners(img)
            t_2_e = time.perf_counter()
            # Make sure OpenCV result is valid
            if not result or len(result) == 0:
                continue
            
            # corners: (N, 4, 2)
            ids, corners = result

            # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            # for corner in corners:
            #     cv2.cornerSubPix(grayscale, corner, (5, 5), (-1, -1), criteria)

            # print(f"Detected corners: {corners}")
            obj_points = []
            image_points = []

            t_3_s = time.perf_counter()
            for i, id in enumerate(ids.flatten()):
                obj_points.extend(tag_points_3D[id])
                image_points.extend(corners[i].reshape(4, 2))
            t_3_e = time.perf_counter()
            marker_data = {}
            
            # Initialize logging variables
            pose_success = False
            raw_rvec_values = ['', '', '']
            raw_tvec_values = ['', '', '']
            filtered_rvec_values = ['', '', '']
            filtered_tvec_values = ['', '', '']
            
            if len(ids) > 0:
                t_4_s = time.perf_counter()
                obj_points = np.array(obj_points, dtype=np.float32)
                image_points = np.array(image_points, dtype=np.float32)

                if last_rvec is not None and last_tvec is not None and last_point_count == len(obj_points):
                    success, rvec, tvec, inliers = cv2.solvePnPRansac(
                        obj_points, 
                        image_points, 
                        camera_matrix, 
                        dist_coeffs, 
                        rvec=last_rvec, 
                        tvec=last_tvec, 
                        useExtrinsicGuess=True,
                        iterationsCount=100,
                        reprojectionError=1.5,
                        confidence=0.95,
                        flags=cv2.SOLVEPNP_ITERATIVE
                    )
                else:
                    success, rvec, tvec = cv2.solvePnP(
                        obj_points,
                        image_points,
                        camera_matrix,
                        dist_coeffs,
                        flags=cv2.SOLVEPNP_SQPNP
                    )

                    inliers = None
                t_4_e = time.perf_counter()
                if success:
                    tvec_mag = np.linalg.norm(tvec)
                    rvec_mag = scipy.spatial.transform.Rotation.from_rotvec(rvec.flatten()).magnitude()

                    if np.any(np.abs(tvec) > 1e5) or np.isnan(tvec).any() or tvec[2] < 0 or sw_t.is_outlier(tvec_mag) or sw_r.is_outlier(rvec_mag):
                        if sw_t.is_outlier(tvec_mag) or sw_r.is_outlier(rvec_mag):
                            print(f"Outlier detected - tvec magnitude: {tvec_mag}, rvec magnitude: {rvec_mag}")

                        print("Pose estimation failed")
                        success = False
                        last_rvec, last_tvec = None, None # Reset for next frame

                        x = np.zeros((16, 1), dtype=np.double)
                        x[10, 0] = 1
                        P = np.eye(15, dtype=np.double) * 0.5
                        continue
                    else:
                        t_5_s = time.perf_counter()
                        # 4. Refine with VVS (Smoother than LM)
                        if inliers is not None and len(inliers) > 0:
                            obj_points_inliers = obj_points[inliers.flatten().astype(int)]
                            image_points_inliers = image_points[inliers.flatten().astype(int)]
                        else:
                            obj_points_inliers = obj_points
                            image_points_inliers = image_points

                        rvec, tvec = cv2.solvePnPRefineVVS(obj_points_inliers, image_points_inliers, camera_matrix, dist_coeffs, rvec, tvec)
                        
                        # Convert received camera pose to OpenCV right-hand coordinates and into scipy/numpy format
                        cam_rot_scipy = scipy.spatial.transform.Rotation.from_quat([-latest_cam_rot[0], latest_cam_rot[1], -latest_cam_rot[2], latest_cam_rot[3]], scalar_first=True)
                        tvec_cam = np.array([tvec[0], tvec[1], tvec[2]])
                        
                        # Tranform to world coordinates
                        tvec_world = [latest_cam_pos[0], -latest_cam_pos[1], latest_cam_pos[2]] + cam_rot_scipy.apply(tvec_cam.flatten())

                        # Transform to world coordinates
                        rvec_scipy = scipy.spatial.transform.Rotation.from_rotvec(rvec.flatten())
                        world_rot = cam_rot_scipy * rvec_scipy
                        rvec_world = world_rot.as_rotvec()

                        
                        rvec_raw = rvec_world.copy()
                        tvec_raw = tvec_world.copy()
                        
                        # Store raw values for CSV logging
                        raw_rvec_values = rvec_raw.flatten().tolist()
                        raw_tvec_values = tvec_raw.flatten().tolist()
                        t_5_e = time.perf_counter()
                        
                        t_6_s = time.perf_counter()
                        # Kalman Filter Predict
                        if (latest_timestamp - previous_timestamp) > 0:
                            dt = (latest_timestamp - previous_timestamp) * 1E-9 # Nanoseconds to seconds
                        else:
                            dt = 0.033
                        
                        kf.predict(dt)

                        # Kalman Filter Update
                        y_k = np.concatenate((tvec_world.flatten(), rvec_world.flatten())).reshape((6, 1))
                        kf.update(y_k)
                        x = kf.x_k
                        dx = kf.dx_k

                        # Symmetrize P to prevent numerical drift
                        P = kf.P_k
                        P = (P + P.T) / 2.0

                        # Add minimum variance floor to diagonal (prevents P collapsing to zero)
                        P += np.eye(15) * 1e-9

                        # Detect divergence before it causes MATLAB warnings
                        if np.linalg.cond(P) > 1e15:
                            print("WARNING: P is ill-conditioned, resetting filter state")
                            x = np.zeros((16, 1), dtype=np.double)
                            x[9, 0] = 1
                            P = np.eye(15) * 0.5
                            last_rvec, last_tvec = None, None
                            previous_timestamp = latest_timestamp
                            continue

                        rotation = scipy.spatial.transform.Rotation.from_quat([x[9, 0], x[10, 0], x[11, 0], x[12, 0]], scalar_first=True)
                        rvec = rotation.as_rotvec().flatten()

                        translation = x[0:3, 0]
                        tvec = translation.reshape((3, 1)).flatten()

                        last_rvec, last_tvec = rvec, tvec

                        t_6_e = time.perf_counter()
                        
                        # Forward projection
                        # Measure latency
                        # t_now_ns = time.time_ns() # TODO: Double check logic, likely different timers
                        # pipeline_latency_s = (t_now_ns - latest_timestamp) * 1e-9
                        pipeline_latency_s = (t_6_e - t_0_s) + 6 * 0.01
                        print(pipeline_latency_s)

                        # Prevent extreme values
                        pipeline_latency_s = np.clip(pipeline_latency_s, 0.0, 0.2)

                        x_forward = kf.future_project(pipeline_latency_s)

                        rotation_forward = scipy.spatial.transform.Rotation.from_quat([x_forward[9, 0], x_forward[10, 0], x_forward[11, 0], x_forward[12, 0]], scalar_first=True)
                        rvec_forward = rotation_forward.as_rotvec().flatten()

                        translation_forward = x_forward[0:3, 0]
                        tvec_forward = translation_forward.reshape((3, 1)).flatten()

                        t_7_s = time.perf_counter()
                        # Mark as successful and store filtered values for CSV logging
                        pose_success = True
                        filtered_rvec_values = rvec_forward.tolist()
                        filtered_tvec_values = tvec_forward.tolist()

                        rvec_kalman = rotation.as_rotvec().flatten()

                        if use_inference:
                            grasped = predictor.is_grasped()
                        else:
                            grasped = False
                            
                        print(f"Grasped: {grasped}")
                        marker_data = {
                            "id": int(ids[0][0]),
                            "tvec": filtered_tvec_values,
                            "rvec": filtered_rvec_values,
                            "grasped": grasped,
                            "timestamp": latest_timestamp
                        }

                        # print("JSON: ", json.dumps(corners_list))
                        # Convert to JSON for sending
                        json_send_message = json.dumps(marker_data) + '\n'
                        # Put message in send queue
                        send_queue.put(json_send_message.encode('utf-8'))
                        t_7_e = time.perf_counter()

                        print(f"T1 (Image Decode): {t_1_e - t_1_s:.4f}s, T2 (Corner Detect): {t_2_e - t_2_s:.4f}s, T3 (PnP Prep): {t_3_e - t_3_s:.4f}s, T4 (PnP Solve): {t_4_e - t_4_s:.4f}s, T5 (VVS Refine): {t_5_e - t_5_s:.4f}s, T6 (KF): {t_6_e - t_6_s:.4f}s, Total: {(t_7_e - t_0_s):.4f}s")
            
            # Write CSV row with timestamp, raw data, filtered data, and pose success flag
            csv_writer.writerow([latest_timestamp, raw_rvec_values[0], raw_rvec_values[1], raw_rvec_values[2], raw_tvec_values[0], raw_tvec_values[1], raw_tvec_values[2], filtered_rvec_values[0], filtered_rvec_values[1], filtered_rvec_values[2], filtered_tvec_values[0], filtered_tvec_values[1], filtered_tvec_values[2], pose_success])
            csv_file.flush()

            previous_timestamp = latest_timestamp
                    
        except Exception as e:
            print(f"Error decoding raw image data: {e}")
            traceback.print_exc()
    
    # Close CSV file on thread exit
    try:
        csv_file.close()
        print(f"CSV file {csv_filename} closed successfully")
    except Exception as e:
        print(f"Error closing CSV file: {e}")
    
    # Close CSV file on thread exit
    try:
        csv_file.close()
        print(f"CSV file {csv_filename} closed successfully")
    except Exception as e:
        print(f"Error closing CSV file: {e}")
    
    # Close CSV file on thread exit
    try:
        csv_file.close()
        print(f"CSV file {csv_filename} closed successfully")
    except Exception as e:
        print(f"Error closing CSV file: {e}")
    
    # --- Business Logic: Enqueue a response if needed ---
    # Example: Echo the message back or send a canned response
    # response = f"Echo: {message}"
    # send_queue.put(response.encode('ascii'))

def receiver_thread(client_socket, addr):
    """Continuously receives data from the client."""
    print(f"Receiver started for {addr}")
    global camera_matrix
    global running
    global latest_timestamp
    global latest_cam_pos
    global latest_cam_rot

    # --- INTRINSICS PHASE ---
    try:
        # Unity sends "fx,fy,cx,cy" as a raw UTF8 string first
        handshake_data = client_socket.recv(1024).decode('utf-8')
        if handshake_data:
            vals = [float(x) for x in handshake_data.split(',')]
            camera_matrix = scale_camera_matrix(vals)
            # e.g., [[428.43286   0.      319.94586]
                   # [  0.      428.43286 320.6509 ]
                   # [  0.        0.        1.     ]]
            print(f"camera intrinsics received:\n{camera_matrix}")
            # Send confirmation back so Unity starts streaming images
            send_queue.put("HANDSHAKE_OK\n".encode('utf-8'))
    except Exception as e:
        print(f"Handshake failed: {e}")
        return
    
    # --- IMAGE RECEIVING PHASE ---
    while running:
        try:
            # print("Waiting to receive length...")

            # Receiver length of incoming image
            length = b''
            while len(length) < 4:
                remaining = client_socket.recv(4 - len(length))

                if not remaining:
                    print("connection closed by client during length reception")
                    break

                length += remaining

            timestamp = b''
            while len(timestamp) < 8:
                remaining = client_socket.recv(8 - len(timestamp))

                if not remaining:
                    print("connection closed by client during timestamp reception")
                    break

                timestamp += remaining
            
            pos_bytes = b''
            while len(pos_bytes) < 12:
                remaining = client_socket.recv(12 - len(pos_bytes))
                if not remaining: break
                pos_bytes += remaining

            rot_bytes = b''
            while len(rot_bytes) < 16:
                remaining = client_socket.recv(16 - len(rot_bytes))
                if not remaining: break
                rot_bytes += remaining

            timestamp = int.from_bytes(timestamp, byteorder='big')

            # print(f"Received length bytes from {str(addr)}: {length}")

            image_size = int.from_bytes(length, byteorder='big')

            cam_pos = np.array(struct.unpack('>fff', pos_bytes), dtype=np.float32)  # x, y, z
            
            cam_rot = np.array(struct.unpack('>ffff', rot_bytes), dtype=np.float32) # w, x, y, z


            # Receive image
            image = b''
            while len(image) < image_size:
                chunk = client_socket.recv(image_size - len(image))

                if not chunk:
                    print("connection closed by client during image reception")
                    break

                image += chunk

            # print(f"Received from {str(addr)} image size {image_size} bytes")

            # Store timestamp globally
            latest_timestamp = timestamp
            latest_cam_pos = cam_pos
            latest_cam_rot = cam_rot

            # If buffer is not empty, replace old data; otherwise, append new data
            if len(latest_image_buffer) > 0:
                latest_image_buffer[0] = image
            else:
                latest_image_buffer.append(image)
            
        except (ConnectionResetError, BrokenPipeError):
            print("Client disconnected.")
            break

        except Exception as e:
            print(f"Receiver error: {e}")
            break
    
    print(f"Receiver for {addr} closing.")

def sender_thread(client_socket, addr):
    """Continuously sends data waiting in the queue to the client."""
    print(f"Sender started for {addr}")

    global running

    while running:
        try:
            # Blocks until an item is available in the queue
            data_to_send = send_queue.get() 
            client_socket.sendall(data_to_send)
            send_queue.task_done()
            
        except queue.Empty:
            continue
            
        except (BrokenPipeError, ConnectionResetError):
            print("Client disconnected (pipe broken).")
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

try:
    while running:
        try:
            # Accept a connection
            client_socket, addr = server_socket.accept()

            # Create a new thread to handle the client
            client_thread = threading.Thread(target=handle_client, args=(client_socket, addr), daemon=True)
            client_thread.start()

        except socket.timeout:
            continue

except Exception as e:
    print(f"program ended: {e}")
    running = False

finally:
    server_socket.close()