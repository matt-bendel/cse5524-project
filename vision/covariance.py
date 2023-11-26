import numpy as np
from tqdm import tqdm
import scipy
import matplotlib.pyplot as plt
import json

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

    def _distance_metric(self, c_model, c_candidate):
        if np.any(np.isnan(c_model)) or np.any(np.isnan(c_candidate)) or np.any(np.isinf(c_model)) or np.any(np.isinf(c_candidate)) or c_model.size == 0 or c_candidate.size == 0:
            print("invalid")
            return 1000
        else:
            e, _ = scipy.linalg.eig(c_model, c_candidate)
            return np.sqrt((np.log(np.abs(e))**2).sum())

    def run(self, initial_bounding_box):
        self.bounding_box = [initial_bounding_box]  # [[bl, br], [tl, tr]]
        x_0, y_0 = initial_bounding_box[0][0]
        x_1, y_1 = initial_bounding_box[1][1]
        frame_t = self.frames[0]
        n_iters = self.frames.shape[0] - 1

        output_file = "covariance_bbox_1-33.json"
        with open(output_file, 'w') as json_file:
            json_file.write("[")

        # for i in tqdm(range(n_iters)):
        for i in tqdm(range(n_iters)):
            frame_t_m_1 = self.frames[i]
            frame_t = self.frames[i+1]
            
            # Speed up the process by assuming the object across the second won't move too far
            start_x, end_x = x_0 - self.window_w // 4, x_1 - self.window_w + self.window_w // 4
            start_y, end_y = y_0 - self.window_h // 4, y_1 - self.window_h + self.window_h // 4

            c_model = self._patch_cov(frame_t_m_1[y_0:y_1, x_0:x_1])
            step = 10 # speed up the scanning process
            match_distances = np.zeros(((end_y - start_y) // step + 1, (end_x - start_x) // step + 1), dtype=float)

            # Assume that the object capture across 10 pixels are not much different to speed up
            for row in range(start_y, end_y, step):
                for col in range(start_x, end_x, step):  
                    window = frame_t[row:row+self.window_h, col:col+self.window_w]
                    c_candidate = self._patch_cov(window)
                    distance = self._distance_metric(c_model, c_candidate)
                    # print(row, col)
                    # print(distance)
                    # print("------------")
                    match_distances[(row - start_y) // step, (col - start_x) // step] = distance


            y, x = np.unravel_index(match_distances.argmin(), match_distances.shape)
            # print(y, x)
            x_0, x_1, y_0, y_1 = x*step + start_x, x*step + start_x + self.window_w, y*step + start_y, y*step + start_y + self.window_h
            # print(y_0, y_1, x_0, x_1)
            bl, br, tl, tr = (x_0, y_0), (x_1, y_0), (x_0, y_1), (x_1, y_1)
            bbox = [[bl, br], [tl, tr]]
            self.bounding_box.append(bbox)

            with open(output_file, 'a') as json_file:
                if i > 0:
                    json_file.write(", ")
                json.dump(str(bbox), json_file)

        with open(output_file, 'a') as json_file:
            json_file.write("]")            

        return self.bounding_box
