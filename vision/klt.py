import cv2
from matplotlib import patches, pyplot as plt
import numpy as np
from tqdm import tqdm
import skimage


class KLTTracker:
    def __init__(self, frames):
        self.frames = frames
        self.bounding_box = []

    def _harris_corner_detection(self, image, threshold=0.01):
        gradient_x = np.gradient(image, axis=1)
        gradient_y = np.gradient(image, axis=0)

        gradient_xx = gradient_x * gradient_x
        gradient_yy = gradient_y * gradient_y
        gradient_xy = gradient_x * gradient_y

        # Define a Gaussian kernel for smoothing
        kernel_size = 3  # Adjust kernel size as needed
        sigma = 1  # Adjust sigma value as needed
        gaussian_kernel = cv2.getGaussianKernel(kernel_size, sigma)
        gaussian_kernel_2d = np.outer(gaussian_kernel, gaussian_kernel.T)

        # Apply Gaussian smoothing to the products of derivatives
        smoothed_xx = cv2.filter2D(gradient_xx, -1, gaussian_kernel_2d)
        smoothed_yy = cv2.filter2D(gradient_yy, -1, gaussian_kernel_2d)
        smoothed_xy = cv2.filter2D(gradient_xy, -1, gaussian_kernel_2d)

        # Calculate the Harris Corner Response function
        det_M = (smoothed_xx * smoothed_yy) - (smoothed_xy ** 2)
        trace_M = smoothed_xx + smoothed_yy
        corner_response = det_M - 0.04 * (trace_M ** 2)

        # Thresholding to find corners
        corner_points = np.zeros_like(corner_response)
        corner_points[corner_response > threshold * corner_response.max()] = 255

        # Extract coordinates of detected corners
        corner_coordinates = np.argwhere(corner_points == 255)

        return corner_coordinates

    def _calculate_optical_flow(self, prev_frame, curr_frame, points):
        new_points = []

        gradient_x_prev = np.gradient(prev_frame, axis=1)
        gradient_y_prev = np.gradient(prev_frame, axis=0)
        gradient_x_curr = np.gradient(curr_frame, axis=1)
        gradient_y_curr = np.gradient(curr_frame, axis=0)

        A = np.zeros((len(points), 2), dtype=np.float64)
        b = np.zeros((len(points), 1), dtype=np.float64)

        for idx, point in enumerate(points):
            x, y = point.astype(np.int32)

            A[idx, 0] = gradient_x_prev[y, x]
            A[idx, 1] = gradient_y_prev[y, x]
            b[idx, 0] = gradient_x_curr[y, x] * gradient_x_prev[y, x] + \
                        gradient_y_curr[y, x] * gradient_y_prev[y, x]
            
        displacement = np.linalg.lstsq(A, b, rcond=None)[0]
        mean_displacement = np.mean(displacement, axis=0)

        new_points = [point + mean_displacement for point in points]

        return np.array(new_points)

    def _update_bbox(self, points):
        # Calculate bounding box from tracked points
        y_0 = int(min(points[:, 0]))
        x_0 = int(min(points[:, 1]))
        y_1 = int(max(points[:, 0]))
        x_1 = int(max(points[:, 1]))

        bl, br, tl, tr = (x_0, y_0), (x_1, y_0), (x_0, y_1), (x_1, y_1)

        return [[bl, br], [tl, tr]]
    
    def run(self, initial_bounding_box):
        self.bounding_box = [initial_bounding_box]  # [[bl, br], [tl, tr]]
        x_0, y_0 = initial_bounding_box[0][0]
        x_1, y_1 = initial_bounding_box[1][1]
        frame_0 = self.frames[0].copy()
        n_iters = self.frames.shape[0] - 1

        initial_patch = frame_0[y_0:y_1, x_0:x_1]
        initial_patch = skimage.color.rgb2gray(initial_patch)
        tracked_points = self._harris_corner_detection(initial_patch) + np.array([y_0, x_0])

        for i in tqdm(range(n_iters)):
            frame_t_m_1 = skimage.color.rgb2gray(self.frames[i])
            frame_t = skimage.color.rgb2gray(self.frames[i+1])

            tracked_points = self._calculate_optical_flow(frame_t_m_1, frame_t, tracked_points)

            bbox = self._update_bbox(tracked_points)
            # print(bbox)
            # x_0, y_0 = initial_bounding_box[0][0]
            # x_1, y_1 = initial_bounding_box[1][1]

            # fig, ax = plt.subplots()
            # ax.imshow(frame_t.astype('uint8'))
            # rect = patches.Rectangle(bbox[0][0], x_1-x_0, y_1-y_0, linewidth=1, edgecolor='r', facecolor='none')
            # ax.add_patch(rect)
            # plt.show()

            self.bounding_box.append(bbox)

        return self.bounding_box
