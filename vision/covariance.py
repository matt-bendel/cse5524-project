import numpy as np
import scipy
import tqdm

class CovarianceTracker:
    def __init__(self, frames, window_h, window_w):
        self.frames = frames
        self.window_h = window_h
        self.window_w = window_w
        self.max_iter = 50 # Speed stuff up
        self.centers = []

    def _patch_cov(self, patch):
        f_k = np.zeros((5, patch.shape[0]*patch.shape[1]))
        for x in range(0, patch.shape[0]):
            for y in range(0, patch.shape[1]):
                f_k[:, (x*patch.shape[1] + y)] = np.array([x, y, patch[x, y, 0], patch[x, y, 1], patch[x, y, 2]])
        return np.cov(f_k, bias=True)

    def _distance_metric(self, c_model, c_candidate):
        e, _ = scipy.linalg.eig(c_model, c_candidate)
        return np.sqrt((np.log(e)**2).sum())

    def run(self, initial_model_xy):
        x_0 = initial_model_xy[0]
        y_0 = initial_model_xy[1]

        self.centers = [(x_0, y_0)]

        n_iters = self.frames.shape[0] - 1

        for i in tqdm(range(n_iters)):
            current_iter = 0
            frame_t_m_1 = self.frames[i]
            frame_t = self.frames[i+1]

            c_model = self._patch_cov(frame_t_m_1[x_0 - self.window_h // 2: x_0 + self.window_h // 2,
                                              y_0 - self.window_w // 2: y_0 + self.window_w // 2])
            match_distances = np.zeros((frame_t_m_1.shape[0] - self.window_h+1, frame_t_m_1.shape[1] - self.window_w+1), dtype=float)
            
            for row in range(frame_t_m_1.shape[0] - self.window_h + 1):
                for col in range(frame_t_m_1.shape[1] - self.window_w + 1):
                    window = frame_t_m_1[row:row+self.window_h, col:col+self.window_w]
                    c_candidate = self._patch_cov(window)
                    distance = self._distance_metric(c_model, c_candidate)
                    match_distances[row, col] = distance

                x, y = np.unravel_index(match_distances.argmin(), match_distances.shape)
                self.centers.append((x+self.window_h, y+self.window_w))

        return self.centers
