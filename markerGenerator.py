import cv2
import numpy as np
import matplotlib.pyplot as plt

arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

markerIds = list(range(0, 6))
markerSize = 50  # pixels

margin = 1
# forming grid
gridHeight = 2 * (markerSize + margin) + margin
gridWidth = 3 * (markerSize + margin) + margin

grid = np.full((gridHeight, gridWidth), 255, dtype=np.uint8)

for id in markerIds:
    row = id // 3
    col = id % 3
    startY = row * (markerSize + margin) + margin
    startX = col * (markerSize + margin) + margin
    grid[startY:(startY + markerSize), startX:(startX + markerSize)] = cv2.aruco.generateImageMarker(arucoDict, id, markerSize)

cv2.imwrite("marker_grid.bmp", grid)

# plt.imshow(markerImage, cmap='gray', interpolation='nearest')
# plt.axis('off')
# plt.title(f'ArUco Marker ID: {markerId}')
# plt.show()