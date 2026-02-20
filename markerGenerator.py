import cv2
import numpy as np
import matplotlib.pyplot as plt

arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

markerIds = list(range(0, 24))
markerSize = 50  # pixels

margin = 27
# forming grid
gridHeight = 4 * (markerSize + margin) + margin
gridWidth = 6 * (markerSize + margin) + margin

grid = np.full((gridHeight, gridWidth), 255, dtype=np.uint8)

for id in markerIds:
    row = id // 6  # 4 rows (0-3)
    col = id % 6   # 6 columns (0-5)
    startY = row * (markerSize + margin) + margin
    startX = col * (markerSize + margin) + margin
    grid[startY:(startY + markerSize), startX:(startX + markerSize)] = cv2.aruco.generateImageMarker(arucoDict, id, markerSize)

cv2.imwrite("marker_grid_24.png", grid, [cv2.IMWRITE_PNG_COMPRESSION, 0])

# plt.imshow(markerImage, cmap='gray', interpolation='nearest')
# plt.axis('off')
# plt.title(f'ArUco Marker ID: {markerId}')
# plt.show()