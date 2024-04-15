import numpy as np
import cv2 as cv
import open3d as o3d

# The given video and calibration data
video_file = './data/chessboard_2x2.mp4'
K = np.array([[696.83969162,  0,            730.33604349],
              [0,             701.95439818, 401.66420193],
              [0,             0,            1]])
dist_coeff = np.array([-0.03263006, 0.04702803,  0.00563099, -0.00420171, 0.02470244])
board_pattern = (9, 7)
board_cellsize = 0.02
board_criteria = cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE + cv.CALIB_CB_FAST_CHECK

# Open a video
video = cv.VideoCapture(video_file)
assert video.isOpened(), 'Cannot read the given input, ' + video_file

# Load the obj model
obj_file = './data/teapot.obj'
mesh = o3d.io.read_triangle_mesh(obj_file)
vertices = np.asarray(mesh.vertices)

mesh.translate((4, 1, 0))

rotation_x = -90 * np.pi / 180
rotation_y = 0 * np.pi / 180
rotation_z = 0 * np.pi / 180
mesh.rotate(mesh.get_rotation_matrix_from_xyz((rotation_x, rotation_y, rotation_z)))

scale_factor = 0.02
vertices *= scale_factor

# Prepare 3D points on a chessboard
obj_points = board_cellsize * np.array([[c, r, 0] for r in range(board_pattern[1]) for c in range(board_pattern[0])])

# Run pose estimation
while True:
    # Read an image from the video
    valid, img = video.read()
    if not valid:
        break

    # Estimate the camera pose
    success, img_points = cv.findChessboardCorners(img, board_pattern, board_criteria)
    if success:
        ret, rvec, tvec = cv.solvePnP(obj_points, img_points, K, dist_coeff)

        # Project the obj model vertices
        proj_vertices, _ = cv.projectPoints(vertices, rvec, tvec, K, dist_coeff)

        # Draw the projected obj model
        triangles = np.asarray(mesh.triangles)
        for triangle in triangles:
            # Get the projected 2D coordinates of the triangle vertices
            p1 = tuple(proj_vertices[triangle[0]][0].astype(int))
            p2 = tuple(proj_vertices[triangle[1]][0].astype(int))
            p3 = tuple(proj_vertices[triangle[2]][0].astype(int))


            # Draw the triangle on the image
            cv.line(img, p1, p2, (255, 0, 0), 1)
            cv.line(img, p2, p3, (255, 0, 0), 1)
            cv.line(img, p3, p1, (255, 0, 0), 1)

        # Print the camera position
        R, _ = cv.Rodrigues(rvec)
        p = (-R.T @ tvec).flatten()
        info = f'XYZ: [{p[0]:.3f} {p[1]:.3f} {p[2]:.3f}]'
        cv.putText(img, info, (10, 25), cv.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0))

    # Show the image and process the key event
    cv.imshow('Pose Estimation (Chessboard)', img)
    key = cv.waitKey(10)
    if key == ord(' '):
        key = cv.waitKey()
    if key == 27:  # ESC
        break

video.release()
cv.destroyAllWindows()
