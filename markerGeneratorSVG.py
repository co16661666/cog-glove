import cv2
import svgwrite  # New dependency

arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

markerIds = list(range(0, 24))
markerSize = 50  # This now acts as "units" (e.g., mm or px)
margin = 27

# Calculate SVG dimensions
gridWidth = 6 * (markerSize + margin) + margin
gridHeight = 4 * (markerSize + margin) + margin

# Initialize SVG Drawing instead of np.full
dwg = svgwrite.Drawing("marker_grid_24.svg", size=(gridWidth, gridHeight), profile='tiny')
dwg.add(dwg.rect(insert=(0, 0), size=(gridWidth, gridHeight), fill='white'))

for id in markerIds:
    row = id // 6
    col = id % 6
    startX = col * (markerSize + margin) + margin
    startY = row * (markerSize + margin) + margin
    
    # Generate the marker bits (dictionary returns 4x4 + 1px border = 6x6 matrix)
    # We use a small size (6) to get the raw bit matrix
    marker_bits = cv2.aruco.generateImageMarker(arucoDict, id, 6)
    pixel_size = markerSize / 6
    
    # Draw each "bit" of the marker as a vector rectangle
    for y in range(6):
        for x in range(6):
            if marker_bits[y, x] == 0:  # If the bit is black
                dwg.add(dwg.rect(
                    insert=(startX + (x * pixel_size), startY + (y * pixel_size)),
                    size=(pixel_size, pixel_size),
                    fill='black'
                ))

# Save the SVG instead of cv2.imwrite
dwg.save()