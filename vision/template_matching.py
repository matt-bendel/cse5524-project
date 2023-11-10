from tqdm import tqdm
import numpy as np


# TODO: Deal with color....
class TemplateMatcher:
    def __init__(self, template):
        self.template = template
        self.template_mean = np.mean(template, axis=(0, 1))
        self.template_std = np.std(template, axis=(0, 1))
        self.template_width = template.shape[1]
        self.template_height = template.shape[0]

    def _ncc(self, P, T, mu_T, sig_T):
        mu_P = np.mean(P, axis=(0, 1))
        sig_P = np.std(P, axis=(0, 1))

        if np.sum(sig_P == 0) > 0 or np.sum(sig_T == 0) > 0:  # Guard against division by zero
            return 0

        standard_P = (P - mu_P[None, None, :]) / sig_P[None, None, :]
        standard_T = (T - mu_T[None, None, :]) / sig_T[None, None, :]

        inner_sum = 1 / (T.shape[0] * T.shape[1] - 1) * np.sum(standard_P * standard_T, axis=(0, 1))

        return np.sum(inner_sum) / 3

    def _get_patch(self, Im, width_shift, height_shift, center):
        return Im[center[1] - height_shift: center[1] + height_shift + 1,
               center[0] - width_shift:center[0] + width_shift + 1, :]

    def run(self, search, start_x, start_y):
        moving_center_y = start_y
        moving_center_x = start_x

        search_width = search.shape[1]
        search_height = search.shape[0]

        ncc_vals = []
        while moving_center_x <= search_width - self.template_width // 2 - 1:
            while moving_center_y <= search_height - self.template_height // 2 - 1:
                # Get patch
                patch = self._get_patch(search, self.template_width // 2, self.template_height // 2,
                                        (moving_center_x, moving_center_y))
                ncc_score = self._ncc(patch, self.template, self.template_mean, self.template_std)

                ncc_vals.append(
                    {
                        'center_x': moving_center_x,
                        'center_y': moving_center_y,
                        'ncc': ncc_score
                    }
                )

                moving_center_y += 1

            moving_center_y = start_y
            moving_center_x += 1

        sorted_vals = sorted(ncc_vals, key=lambda x: x['ncc'], reverse=True)
        best_match_center = (sorted_vals[0]['center_x'], sorted_vals[0]['center_y'])

        return best_match_center
