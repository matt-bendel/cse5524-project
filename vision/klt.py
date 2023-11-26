import math
import numpy as np
from tqdm import tqdm
import scipy
import scipy.signal as signal
from scipy.ndimage import gaussian_filter
import cv2
import matplotlib.pyplot as plt

class KLTTracker:
    def __init__(self, frames, window_h, window_w):
        self.frames = frames
        self.window_h = window_h
        self.window_w = window_w
        self.optical_flow = LKOpticalFlow()
        self.bounding_box = []

    def run(self, initial_bounding_box):
        num_frames = len(self.frames)
        initial_frame = cv2.cvtColor(self.frames[0].astype('uint8'), cv2.COLOR_BGR2GRAY)
        
        initial_keypoints = self.select_initial_keypoints(initial_frame, initial_bounding_box)
        self.bounding_box.append(self.get_bounding_box(initial_keypoints))

        for i in range(num_frames - 1):
            current_frame = cv2.cvtColor(self.frames[i].astype('uint8'), cv2.COLOR_BGR2GRAY)
            next_frame = cv2.cvtColor(self.frames[i + 1].astype('uint8'), cv2.COLOR_BGR2GRAY)

            tracked_points = self.optical_flow.compute_optical_flow(current_frame, next_frame, initial_keypoints, self.window_h, self.window_w)
            plt.imshow(tracked_points)
            plt.show()
            self.bounding_box.append(self.get_bounding_box(tracked_points))

        return self.bounding_box

    def select_initial_keypoints(self, frame, initial_bounding_box):
        self.bounding_box = [initial_bounding_box]
        x_0, y_0 = initial_bounding_box[0][0]
        x_1, y_1 = initial_bounding_box[1][1]
        
        window = frame[y_0:y_1, x_0:x_1]
        points = self.detect_features(window)
        initial_keypoints = points + np.array([x_0, y_1])

        return initial_keypoints

    def detect_features(self, frame):
        
        def gaussDeriv2D(sigma):
            # Set mask size
            x = np.arange(-math.ceil(3*sigma), math.ceil(3*sigma)+1, 1)
            y = np.arange(-math.ceil(3*sigma), math.ceil(3*sigma)+1, 1)

            Gx = np.zeros((y.size, x.size))
            Gy = np.zeros((y.size, x.size))
            G = np.zeros((y.size, x.size))

            for i in range(y.size):
                for j in range(x.size):
                    Gx[i][j] = (x[j] / (2*np.pi*sigma**4)) * np.exp(-(x[j]**2 + y[i]**2)/(2*sigma**2))
                    Gy[i][j] = (y[i] / (2*np.pi*sigma**4)) * np.exp(-(x[j]**2 + y[i]**2)/(2*sigma**2))
                    G[i][j] = (1 / (2*np.pi*sigma**2)) * np.exp(-(x[j]**2 + y[i]**2)/(2*sigma**2))

            return Gx, Gy, G

        # Gx, Gy gradients
        Gx, Gy, _ = gaussDeriv2D(sigma=0.7)
        # Gaussian window
        G = gaussDeriv2D(sigma=1)[2]
        # Normalize smoothing mask and abs derivative masks
        Gx /= np.abs(Gx).sum()
        Gy /= np.abs(Gy).sum()
        G /= G.sum()

        I_x = scipy.ndimage.convolve(frame, Gx)
        I_y = scipy.ndimage.convolve(frame, Gy)
        I_xx = I_x**2
        I_yy = I_y**2
        I_xy = I_x*I_y

        gI_xx = scipy.ndimage.convolve(I_xx, G)
        gI_yy = scipy.ndimage.convolve(I_yy, G)
        gI_xy = scipy.ndimage.convolve(I_xy, G)

        R = gI_xx*gI_yy - gI_xy**2 - 0.05*(gI_xx+gI_yy)**2
        R -= 200
        R[R < 0] = 0 
        
        def non_maximum_suppression(R, neighborhood_size=10):
            threshold = 0.01
            corner_points = []
            half_size = neighborhood_size // 2
            for row in range(half_size, R.shape[0] - half_size):
                for col in range(half_size, R.shape[1] - half_size):
                    if R[row, col] > threshold:
                        # extract a 3x3 region
                        local_region = R[row - half_size:row + half_size + 1, col - half_size:col + half_size + 1]
                        max_in_region = local_region.max()
                        max_coords = np.argwhere(local_region == max_in_region)
                        # ensure max and uniqueness
                        if R[row, col] == max_in_region and len(max_coords) == 1:
                            corner_points.append((row, col))
            return corner_points

        corner_points = non_maximum_suppression(R)
        corner_points = np.array(corner_points)

        return corner_points

    def get_bounding_box(self, points):
        min_x = int(min(points[:, 1]))
        min_y = int(min(points[:, 0]))
        max_x = int(max(points[:, 1]))
        max_y = int(max(points[:, 0]))

        bl, br, tl, tr = (min_x, min_y), (max_x, min_y), (min_x, max_y), (max_x, max_y)
        bbox = [[bl, br], [tl, tr]]
        print(bbox)

        return bbox

class LKOpticalFlow:
    def __init__(self):
        self.max_iters = 30
        self.eps = 0.03

    def compute_optical_flow(self, current_frame, next_frame, keypoints, window_h, window_w):
        updated_keypoints = np.array(keypoints).astype(float)
        
        for _ in range(self.max_iters):
            u, v = np.zeros_like(updated_keypoints[:, 0], dtype=float), np.zeros_like(updated_keypoints[:, 1], dtype=float)

            for i, (x, y) in enumerate(updated_keypoints):
                x = int(x)
                y = int(y)
                if y - 1 >= 0 and y + 2 <= current_frame.shape[0] and x - 1 >= 0 and x + 2 <= current_frame.shape[1]:
                    patch1 = current_frame[y - 1:y + 2, x - 1:x + 2]
                    patch2 = next_frame[y - 1:y + 2, x - 1:x + 2]

                flow_u, flow_v = self.optical_flow(patch1, patch2)
                
                u[i] += flow_u[1, 1]
                v[i] += flow_v[1, 1]

            updated_keypoints[:, 0] += u.flatten()
            updated_keypoints[:, 1] += v.flatten()
        
        return updated_keypoints

    def optical_flow(self, patch1, patch2):
        kernel_x = 0.25 * np.array([[-1., 1.], [-1., 1.]])
        kernel_y = 0.25 * np.array([[-1., -1.], [1., 1.]])
        kernel_t = 0.25 * np.array([[1., 1.], [1., 1.]])
        kernel_x = np.fliplr(kernel_x)
        mode = 'same'
        fx = (signal.convolve2d(patch1, kernel_x, boundary='symm', mode=mode))
        fy = (signal.convolve2d(patch1, kernel_y, boundary='symm', mode=mode))
        ft = (signal.convolve2d(patch2, kernel_t, boundary='symm', mode=mode) + 
             signal.convolve2d(patch1, -kernel_t, boundary='symm', mode=mode))
    
        u = np.zeros(patch1.shape)
        v = np.zeros(patch1.shape)
    
        window = np.ones((3, 3))
        denom = (signal.convolve2d(fx**2, window, boundary='symm', mode=mode) * 
                 signal.convolve2d(fy**2, window, boundary='symm', mode=mode) -
                 signal.convolve2d(fx*fy, window, boundary='symm', mode=mode)**2)
        denom[denom == 0] = 1
    
        u = ((signal.convolve2d(fy**2, window, boundary='symm', mode=mode) * 
              signal.convolve2d(fy*ft, window, boundary='symm', mode=mode) + 
              signal.convolve2d(fx*fy, window, boundary='symm', mode=mode) * 
              signal.convolve2d(fy*ft, window, boundary='symm', mode=mode)) /
             denom)
    
        v = ((signal.convolve2d(fx*ft, window, boundary='symm', mode=mode) * 
              signal.convolve2d(fx*fy, window, boundary='symm', mode=mode) -
              signal.convolve2d(fx**2, window, boundary='symm', mode=mode) * 
              signal.convolve2d(fy*ft, window, boundary='symm', mode=mode)) / 
             denom)

        return (u, v)
