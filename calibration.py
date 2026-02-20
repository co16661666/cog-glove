import numpy as np
import cv2 as cv
import glob

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points for 7x4 chessboard (7 columns, 4 rows)
# Points range from (0,0,0) to (6,3,0) = 28 total inner corners
objp = np.zeros((4*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:4].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = sorted(glob.glob('image_*.jpg'))[::20]

print(f"Found {len(images)} images")
print("Looking for 7x4 chessboard pattern (7 columns x 4 rows = 28 inner corners)...")
print("\nNOTE: If no chessboards are found, common reasons are:")
print("  1. Images contain ArUco markers instead of chessboards")
print("  2. Chessboard not fully visible in frame")
print("  3. Wrong pattern size (will try multiple sizes)")
print("  4. Poor lighting or contrast")
print("  5. Image blur or motion")
print()

for fname in images:
    img = cv.imread(fname)
    if img is None:
        print(f"Failed to load {fname}")
        continue
        
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    # Try different pattern sizes in case the chessboard size is wrong
    # Format: (columns, rows) - inner corners
    # pattern_sizes = [(7,4), (8,5), (6,4), (7,5), (8,4), (6,5), (9,6), (7,6), (6,6), (5,4), (8,6)]
    pattern_sizes = [(8,5)]
    found = False
    
    for pattern_size in pattern_sizes:
        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, pattern_size, None)
        
        if ret == True:
            # Use the correct objp for this pattern size
            cols, rows = pattern_size
            temp_objp = np.zeros((rows*cols, 3), np.float32)
            temp_objp[:,:2] = np.mgrid[0:cols, 0:rows].T.reshape(-1,2)
            
            objpoints.append(temp_objp)
            corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners2)
            
            # Draw and display the corners
            cv.drawChessboardCorners(img, pattern_size, corners2, ret)
            cv.imshow('Chessboard Detected', img)
            cv.waitKey(1)
            print(f"✓ {fname}: Found {cols}x{rows} chessboard")
            found = True
            break
    
    if not found:
        print(f"✗ {fname}: Chessboard not found")
        # Show the image so user can see what's actually in it
        cv.imshow('No pattern found - check image', img)
        cv.waitKey(2)

cv.destroyAllWindows()

print(f"\nTotal images with chessboard detected: {len(objpoints)}")

# Perform camera calibration
if len(objpoints) > 0:
    print(f"\nCalibrating camera with {len(objpoints)} images...")
    
    # Get image size from the last processed image
    # gray.shape[::-1] gives (width, height)
    img_size = gray.shape[::-1]
    
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, img_size, None, None)
    
    print("\n=== Calibration Results ===")
    print(f"Reprojection error: {ret:.4f} pixels")
    print(f"\nCamera Matrix (mtx):")
    print(mtx)
    print(f"\nDistortion Coefficients (dist):")
    print(dist)
    
    # Save calibration results
    np.savez('calibration_data.npz', 
             camera_matrix=mtx, 
             dist_coeffs=dist,
             rvecs=rvecs,
             tvecs=tvecs)
    print("\n✓ Calibration data saved to 'calibration_data.npz'")
else:
    print("\n✗ Error: No chessboard patterns found in any images!")
    print("   Cannot perform calibration without detected patterns.")