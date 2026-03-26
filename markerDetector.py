import cv2
import numpy as np

# # Load the image
# image = cv2.imread('marker_screenshot.png')

def getCorners(image):
    if image is None:
        return []
    
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    
    parameters = cv2.aruco.DetectorParameters()

    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX # Also try CORNER_REFINE_CONTOUR
    # parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
    # parameters.cornerRefinementWinSize = 2
    # parameters.cornerRefinementMinAccuracy = 0.05
    # parameters.cornerRefinementMaxIterations = 30 # default

    # Create the ArUco detector
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    # Detect the markers
    corners, ids, rejected = detector.detectMarkers(image)

    # Print the detected markers
    # print("Detected markers:", ids)
    if ids is not None:
        # cv2.aruco.drawDetectedMarkers(image, corners, ids)
        # cv2.imshow('Detected Markers', image)
        # cv2.waitKey(1)

        return ids, corners
    # else:
    #     cv2.imshow('Detected Markers', image)
    #     cv2.waitKey(1)

        return []