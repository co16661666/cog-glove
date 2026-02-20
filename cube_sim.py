import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

    rot_mat = get_rotation_matrix_90(rot_axis, rot_amount).T
    tag_points_3D[i + 0] = template_tag[(i + 0) % 4] @ rot_mat + offset
    tag_points_3D[i + 1] = template_tag[(i + 1) % 4] @ rot_mat + offset
    tag_points_3D[i + 2] = template_tag[(i + 2) % 4] @ rot_mat + offset
    tag_points_3D[i + 3] = template_tag[(i + 3) % 4] @ rot_mat + offset

def plot_cube_tags(tag_points_3D):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 6 colors for the 6 faces of the cube
    colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown']
    face_names = ["Top", "Back", "Left", "Front", "Right", "Bottom"]

    # Iterate through the 6 faces (each face contains 4 tags)
    for face_idx in range(6):
        color = colors[face_idx]
        
        # Iterate through the 4 tags belonging to this face
        for tag_in_face in range(4):
            tag_id = face_idx * 4 + tag_in_face
            
            # Extract the 4 corners of the tag
            # We add the first point to the end to close the square loop
            tag_pts = tag_points_3D[tag_id]
            loop_pts = np.vstack([tag_pts, tag_pts[0]]) 

            # Plot the tag outline
            ax.plot(loop_pts[:, 0], loop_pts[:, 1], loop_pts[:, 2], 
                    color=color, linewidth=1.5, alpha=0.8)
            
            # Optional: Label each tag ID at its center
            center = np.mean(tag_pts, axis=0)
            ax.text(center[0], center[1], center[2], str(tag_id), 
                    color=color, fontsize=8)

        # Add a large label for the Face name at the average center of its 4 tags
        all_face_pts = np.vstack([tag_points_3D[i] for i in range(face_idx*4, (face_idx+1)*4)])
        face_center = np.mean(all_face_pts, axis=0)
        ax.text(face_center[0], face_center[1], face_center[2], face_names[face_idx], 
                color='black', fontsize=12, fontweight='bold', 
                bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))

    # Equalize axes so it looks like a cube
    max_range = 0.04 # Based on your ~0.02 coordinates
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_zlim(-max_range, max_range)
    
    ax.set_xlabel('X (Side to Side)')
    ax.set_ylabel('Y (Top to Bottom)')
    ax.set_zlabel('Z (Front to Back)')
    ax.set_box_aspect([1,1,1])
    
    plt.title("3D Tag Configuration (6 Faces, 4 Tags Each)")
    plt.show()

# Call this after your loop
plot_cube_tags(tag_points_3D)