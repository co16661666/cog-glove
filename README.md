# Cognition Glove OpenCV Pose Estimation
This repo is a part of the cognition glove project I am working on that involves pose estimation of a cube from camera data captured on a Meta Quest 3/3S. It is designed to work with the [frontend of this project](https://github.com/co16661666/cog-glove-passthrough), which is a heavily modified fork of [Meta's Unity-PassthroughCameraAPISamples repo](https://github.com/oculus-samples/Unity-PassthroughCameraApiSamples). This project connects to the frontend using Android Debug Bridge (ADB) port forwarding and a TCP communication protocol for sending images and pose data. Overall, this backend repo encompasses 

In addition to pose detection, this project also involves grasp detection of the cube through a glove with embedded sensors. The glove sends sensor data over Serial port, and graspInference.py runs a binary classification algorithm to determine grasp success.
## Running the Program
**The main file is `imageTCP.py`**, the remaining scripts are mainly helper functions and classes. onnx_classifier_test.py allows testing of the classification model.
## TCP Communication Details
### Handshake Phase
After the initial connection, but before image streaming begins, imageTCP.py expects to receive the camera intrinsics as a UTF-8 string formatted as `{fx},{fy},{cx},{cy},{targetWidth},{targetHeight}`. Currently, the variables are sent as floats, but the Python TCP server reads up to 1024 bytes. The script will then send a confirmation string  `HANDSHAKE_OK\n` (also UTF-8 encoded) to begin image streaming.
### Image Streaming (Headset -> Computer)
Image streaming can be broken down into 6 parts:
1. **Image Length:** Length of incoming image in bytes                                                  | 4 Bytes (int)
2. **Image Timestamp:** Timestamp of incoming image                                                     | 8 Bytes (float)
3. **Position of Cube:** Vector position of cube at time of image capture (Headset frame of reference)  | 12 Bytes (3 floats)
4. **Rotation of Cube:** Quaternion rotation of cube at time of image capture (Headset frame)           | 16 Bytes (4 floats)
5. **Image:** Image from passthrough                                                                    | Image Length (Expected 720x720 px)

The image format is expected to be 720x720 px, although in the future, it is planned for it to be received alongside the camera intrinsics. Additionally, the image color format should be R8 (grayscale) to reduce overhead, and OpenCV does not benefit from colored images for ArUco detection.
### Pose Streaming (Computer -> Headset)
The imageTCP.py script actively grabs the most recent image from the image buffer, running on a separate thread, to perform pose estimation. The steps are as follow:
1. OpenCV pipeline (corner detection + pose estimation)
2. Kalman filtering
3. Check classification model for pose
4. Send filtered pose (position and rotation) back to Unity TCP client

The data is sent back to the Quest as a UTF-8 encoded JSON:
```
marker_data = {
    "id": int(ids[0][0]),
    "tvec": filtered_tvec_values,
    "rvec": filtered_rvec_values,
    "grasped": grasped,
    "timestamp": latest_timestamp
}
```
where `id` (currently unused) is a list of the IDs of the markers detected; `tvec` is the translation vector (OpenCV frame); `rvec` is the rotation vector (OpenCV frame); `grasped` is whether or not the cube is classified as grasped; and `timestamp` is the timestamp of the image for lookup on Unity side.
## Pose Estimation Pipeline
The pose estimation involves OpenCV corner detection and solvePnP, which is then passed through an error-state extended Kalman filter (ESEKF)
### OpenCV Pipepline
#### Corner Detection
Key details:
- File: `markerDetector.py`
- Marker dictionary: `DICT_4X4_50`
- Corner refinement method: `CORNER_REFINE_SUBPIX`
#### Pose Estimation
Key details:
- `solvePnPRansac` with `SOLVEPNPITERATIVE` flag
- Fallback to `solvePnP` with `SOLVEPNP_SQPNP` if last solve failed
### Refinement
Key details:
- solvePnPRefineVVS
## Error-State Extended Kalman Filter
Kalman filter class is in `OpenCV_KF.py` and implemented in `imageTCP.py`. It uses a model assuming constant accleration and constant rotational velocity.
