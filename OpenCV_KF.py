import numpy as np
from scipy.spatial.transform import Rotation as Rot

class OpenCV_KF:
    def __init__(self, x_k, dx_k, P_k, Q, R):
        self.x_k = x_k  # State vector
        self.dx_k = dx_k  # State transition matrix
        self.P_k = P_k  # Estimate error covariance
        self.Q = Q  # Covariance of process noise constant
        self.R = R  # Covariance of measurement noise

    # ========== PREDICTION STEP ==========
    def predict(self, dt):
        # STATE PREDICTION
        # Position
        self.x_k[0:3, 0] = self.x_k[0:3, 0] + self.x_k[3:6, 0] * dt + 0.5 * self.x_k[6:9, 0] * dt ** 2
        self.x_k[3:6, 0] = self.x_k[3:6, 0] + self.x_k[6:9, 0] * dt
        # self.x_k[6:9, 0] = self.x_k[6:9, 0]

        # Rotation
        q_k = Rot.from_quat([self.x_k[9, 0], self.x_k[10, 0], self.x_k[11, 0], self.x_k[12, 0]], scalar_first=True)
        wt = Rot.from_rotvec(self.x_k[13:16, 0] * dt)
        rotation = Rot.as_quat(wt * q_k, scalar_first=True)
        self.x_k[9:13, 0] = [rotation[0], rotation[1], rotation[2], rotation[3]]
        # self.x_k[13:16, 0] = self.x_k[13:16, 0]

        # ERROR PREDICTION
        # Define F and L matrices
        F_k = np.block([
            [np.eye(3, 3), np.eye(3, 3) * dt, 0.5 * np.eye(3, 3) * dt ** 2, np.zeros((3, 3)), np.zeros((3, 3))],
            [np.zeros((3, 3)), np.eye(3, 3), np.eye(3, 3) * dt, np.zeros((3, 3)), np.zeros((3, 3))],
            [np.zeros((3, 3)), np.zeros((3, 3)), np.eye(3, 3), np.zeros((3, 3)), np.zeros((3, 3))],
            [np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3)), np.eye(3, 3), np.eye(3, 3) * dt],
            [np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3)), np.eye(3, 3)]
        ])

        L_k = np.zeros((15, 6))
        L_k[6:9, 0:3] = np.eye(3, 3)
        L_k[12:15, 3:6] = np.eye(3, 3)

        # Predicted error update
        Q_k = self.Q * dt
        self.P_k = F_k @ self.P_k @ np.transpose(F_k) + L_k @ Q_k @ np.transpose(L_k)

    # ========== PREDICTION STEP ==========
    def update(self, y_k):
        # Define H matrix
        H_k = np.zeros((6, 15))
        H_k[0:3, 0:3] = np.eye(3, 3)
        H_k[3:6, 9:12] = np.eye(3, 3)
        
        #Define R matrix
        R_k = self.R

        # Calculate Kalman gain
        S = (H_k @ self.P_k @ np.transpose(H_k) + R_k)
        K_k = self.P_k @ np.transpose(H_k) @ np.linalg.inv(S)

        # Calculate difference between measured and predicted
        z_k = np.zeros((6, 1))
        z_k[0:3, 0] = y_k[0:3, 0] - self.x_k[0:3, 0]
        
        rot_y = Rot.from_rotvec([y_k[3, 0], y_k[4, 0], y_k[5, 0]])
        rot_k = Rot.from_quat([self.x_k[9, 0], self.x_k[10, 0], self.x_k[11, 0], self.x_k[12, 0]], scalar_first=True)
        z_k[3:6, 0] = np.transpose((rot_y * rot_k.inv()).as_rotvec())

        # Update error state
        self.dx_k = K_k @ z_k

        # Update state error
        self.P_k = (np.eye(15, 15) - K_k @ H_k) @ self.P_k @ np.transpose(np.eye(15, 15) - K_k @ H_k) + K_k @ R_k @ np.transpose(K_k)

        # Update state vector
        self.x_k[0:9, 0] = self.x_k[0:9, 0] + self.dx_k[0:9, 0]

        q_k = Rot.from_quat([self.x_k[9, 0], self.x_k[10, 0], self.x_k[11, 0], self.x_k[12, 0]], scalar_first=True)
        dq = Rot.from_rotvec([self.dx_k[9, 0], self.dx_k[10, 0], self.dx_k[11, 0]])
        self.x_k[9:13, 0] = np.transpose((dq * q_k).as_quat(scalar_first=True))

        self.x_k[13:16, 0] = self.x_k[13:16, 0] + self.dx_k[12:15, 0]
        # rvec = rotation.as_rotvec().flatten()

        self.dx_k = np.zeros((15, 1))

