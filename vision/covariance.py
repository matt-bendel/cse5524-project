import numpy as np
from tqdm import tqdm
import scipy

class CovarianceTracker:
    def __init__(self, frames, window_h, window_w):
        self.frames = frames
        self.window_h = window_h
        self.window_w = window_w
        self.bounding_box = []

    def _patch_cov(self, patch):
        f_k = np.zeros((5, patch.shape[0]*patch.shape[1]))
        for x in range(0, patch.shape[0]):
            for y in range(0, patch.shape[1]):
                f_k[:, (x*patch.shape[1] + y)] = np.array([x, y, patch[x, y, 0], patch[x, y, 1], patch[x, y, 2]])
        return np.cov(f_k, bias=True)

        # x, y = np.meshgrid(range(patch.shape[0]), range(patch.shape[1]), indexing='ij')
        # f_k = np.vstack((x.flatten(), y.flatten(), patch[:, :, 0].flatten(), patch[:, :, 1].flatten(), patch[:, :, 2].flatten()))
        # return np.cov(f_k, bias=True)

    def _distance_metric(self, c_model, c_candidate):
        e, _ = scipy.linalg.eig(c_model, c_candidate)
        return np.sqrt((np.log(np.abs(e))**2).sum())

    def run(self, initial_bounding_box):
        self.bounding_box = [initial_bounding_box]  # [[bl, br], [tl, tr]]
        x_0, y_0 = initial_bounding_box[0][0]
        x_1, y_1 = initial_bounding_box[1][0]

        n_iters = self.frames.shape[0] - 1

        for i in tqdm(range(n_iters)):
            frame_t_m_1 = self.frames[i]
            frame_t = self.frames[i+1]

            c_model = self._patch_cov(frame_t_m_1[y_0:y_1, x_0:x_1])
            match_distances = np.zeros((frame_t_m_1.shape[0] - self.window_h+1, frame_t_m_1.shape[1] - self.window_w+1), dtype=float)
            
            total_iterations = ((frame_t.shape[0] - self.window_h) // 10 + 1) * ((frame_t.shape[1] - self.window_w) // 10 + 1)
            pbar = tqdm(total=total_iterations, desc='Processing')

            # for row in range(frame_t.shape[0] - self.window_h + 1):
            #     for col in range(frame_t.shape[1] - self.window_w + 1):
            for row in range(0, frame_t.shape[0] - self.window_h + 1, 10):
                for col in range(0, frame_t.shape[1] - self.window_w + 1, 10):
                    window = frame_t[row:row+self.window_h, col:col+self.window_w]
                    c_candidate = self._patch_cov(window)
                    distance = self._distance_metric(c_model, c_candidate)
                    match_distances[row, col] = distance

                    pbar.update(1)

            pbar.close()

            x, y = np.unravel_index(match_distances.argmin(), match_distances.shape)
            x_0, x_1, y_0, y_1 = x, x+self.window_w, y, y+self.window_h 
            bl, br, tl, tr = (x_0, y_0), (x_1, y_0), (x_0, y_1), (x_1, y_1)
            self.bounding_box.append([[bl, br], [tl, tr]])

        return self.bounding_box
