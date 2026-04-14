# Cognition Glove OpenCV Pose Estimation
This repo is a part of the cognition glove project I am working on that involves pose estimation of a cube from camera data captured on a Meta Quest 3/3S. It is designed to work with the [frontend of this project](https://github.com/co16661666/cog-glove-passthrough), which is a heavily modified fork of [Meta's Unity-PassthroughCameraAPISamples repo](https://github.com/oculus-samples/Unity-PassthroughCameraApiSamples). This project connects to the frontend using Android Debug Bridge (ADB) port forwarding and a TCP communication protocol for sending images and pose data. Overall, this backend repo encompasses 

In addition to pose detection, this project also involves grasp detection of the cube through a glove with embedded sensors. The glove sends sensor data over Serial port, and graspInference.py runs a binary classification algorithm to determine grasp success.
## Running the Program
The main file is imageTCP.py, the remaining scripts are mainly helper functions and classes. onnx_classifier_test.py allows testing of the classification model.
## TCP Communication Details
### Handshake Protocol
After the initial connection, but before image streaming begins, imageTCP.py expects to receive the camera intrinsics as a UTF-8 string formatted as `{fx},{fy},{cx},{cy},{targetWidth},{targetHeight}`. Currently, the variables are sent as floats, but the Python TCP server reads up to 1024 bytes. The script will then send a confirmation string  `HANDSHAKE_OK\n` (also UTF-8 encoded) to begin image streaming.
