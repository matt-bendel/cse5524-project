import numpy as np

class KLTTracker:
    def __init__(self, frames):
        self.frames = frames
        self.optical_flow = LKOpticalFlow(frames)  # Initialize the LKOpticalFlow
        self.tracked_keypoints = []
        self.window_size = 5  # Define your window size for tracking here

    def detect_keypoints(self, frame):
        # Implement a method to detect keypoints in a frame
        # For example, using corner detection methods like Harris Corner Detector or Shi-Tomasi
        # Here, I'm assuming a simple random choice of keypoints for demonstration purposes
        height, width, _ = frame.shape
        num_keypoints = 10
        x = np.random.randint(0, width, num_keypoints)
        y = np.random.randint(0, height, num_keypoints)
        return np.array(list(zip(x, y)))

    def track_keypoints(self):
        num_frames = len(self.frames)
        for i in range(num_frames - 1):
            current_frame = self.frames[i]
            next_frame = self.frames[i + 1]

            if i == 0:
                self.optical_flow.compute_optical_flow()  # Compute once for initial keypoints

            tracked_points = self.optical_flow.compute_optical_flow()
            self.tracked_keypoints.append(tracked_points)

        return self.tracked_keypoints

class LKOpticalFlow:
    def __init__(self, frames):
        self.frames = frames
        self.window_size = 5  # Define your window size for tracking here
        self.num_levels = 3  # Pyramid levels for coarse-to-fine approach
        self.threshold = 0.01  # Convergence threshold
        self.max_iters = 50  # Maximum iterations for each point

    def compute_gradients(self, frame):
        # Compute gradients using Sobel operators
        gx = np.gradient(frame, axis=1)
        gy = np.gradient(frame, axis=0)
        return gx, gy

    def compute_optical_flow(self):
        num_frames = len(self.frames)
        tracked_points = []

        for i in range(num_frames - 1):
            current_frame = self.frames[i]
            next_frame = self.frames[i + 1]

            # Detect key points (for simplicity, using random points here)
            height, width, _ = current_frame.shape
            num_keypoints = 10
            x = np.random.randint(self.window_size, width - self.window_size, num_keypoints)
            y = np.random.randint(self.window_size, height - self.window_size, num_keypoints)
            points = np.array(list(zip(x, y)), dtype=np.float32)

            for level in range(self.num_levels, 0, -1):
                for j, (x, y) in enumerate(points):
                    x = int(x)
                    y = int(y)

                    # Define the window around the point
                    window_x = current_frame[max(0, y - self.window_size):min(y + self.window_size, height),
                                             max(0, x - self.window_size):min(x + self.window_size, width)]

                    gx, gy = self.compute_gradients(window_x)

                    # Compute the gradients in the window
                    A = np.column_stack((gx.flatten(), gy.flatten()))

                    b = -np.dot(A.T, np.ones(A.shape[0]))

                    delta_p = np.zeros(2)
                    delta = np.inf

                    for _ in range(self.max_iters):
                        x_warp = int(x + delta_p[0])
                        y_warp = int(y + delta_p[1])

                        window_next = next_frame[max(0, y_warp - self.window_size):min(y_warp + self.window_size, height),
                                                 max(0, x_warp - self.window_size):min(x_warp + self.window_size, width)]

                        error = window_x - window_next

                        error = error.flatten()

                        # Compute error gradient
                        error_grad = np.dot(A.T, error)

                        # Compute the update to the parameter
                        dp = np.linalg.lstsq(A.T @ A, -A.T @ error, rcond=None)[0]

                        delta_p += dp

                        # Check convergence
                        delta = np.linalg.norm(dp)
                        if delta < self.threshold:
                            break

                    points[j] += delta_p

                points /= 2  # Downsample the points for the next pyramid level

            tracked_points.append(points)

        return tracked_points
