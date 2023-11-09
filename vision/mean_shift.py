import numpy as np

class MeanShiftTracker:
    def __init__(self, frames, n_bins, kernel_h, eps=1e-2):
        self.frames = frames
        self.n_bins = n_bins
        self.kernel_h = kernel_h
        self.eps = eps
        self.centers = []

    def _circular_neighbors(self, img, x, y, radius, get_coords=False):
        h = img.shape[0]
        w = img.shape[1]

        # Center values
        x_coord = np.arange(w) - x
        y_coord = np.arange(h) - y

        coords = np.zeros((2, h, w))
        coords[0, :, :] = np.repeat(np.expand_dims(x_coord, axis=0), h, axis=0)  # x values
        coords[1, :, :] = np.repeat(np.expand_dims(y_coord, axis=1), w, axis=1)  # y values

        circle_eq = coords[0, :, :] ** 2 + coords[1, :, :] ** 2 - radius ** 2

        if get_coords:
            circle_eq = np.repeat(np.expand_dims(circle_eq, axis=2), 3, axis=2)
            return np.where(circle_eq > 0, 0, img)

        inds = np.where(circle_eq <= 0)

        feats = img[inds[0], inds[1], :]
        augmented_features = np.zeros((feats.shape[0], 5))
        augmented_features[:, 0] = inds[1]  # coords[0, inds[0], inds[1]]
        augmented_features[:, 1] = inds[0]  # coords[1, inds[0], inds[1]]
        augmented_features[:, 2:] = feats

        return np.reshape(augmented_features, (-1, 5))

    def _color_histogram(self, X, bins, x, y, h):
        hist = np.zeros((bins, bins, bins))
        bin_ranges = [(i * 256 // bins, (i + 1) * 256 // bins - 1) for i in range(bins)]

        for feat in X:
            bin_inds = [-1, -1, -1]
            for i, bin_range in enumerate(bin_ranges):
                if bin_range[1] >= feat[2] >= bin_range[0]:
                    bin_inds[0] = i

                if bin_range[1] >= feat[3] >= bin_range[0]:
                    bin_inds[1] = i

                if bin_range[1] >= feat[4] >= bin_range[0]:
                    bin_inds[2] = i

            r = (np.sqrt((feat[0] - x) ** 2 + (feat[1] - y) ** 2) / h) ** 2
            k = np.where(r < 1, 1 - r, 0)
            hist[bin_inds[0], bin_inds[1], bin_inds[2]] += k

        return hist / np.sum(hist)

    def _get_weights(self, X, q_model, p_test, bins):
        bin_ranges = [(i * 256 // bins, (i + 1) * 256 // bins - 1) for i in range(bins)]
        w = np.zeros((X.shape[0]))

        for j, feat in enumerate(X):
            bin_inds = [-1, -1, -1]
            for i, bin_range in enumerate(bin_ranges):
                if bin_range[1] >= feat[2] >= bin_range[0]:
                    bin_inds[0] = i

                if bin_range[1] >= feat[3] >= bin_range[0]:
                    bin_inds[1] = i

                if bin_range[1] >= feat[4] >= bin_range[0]:
                    bin_inds[2] = i

            w[j] += np.sqrt(
                q_model[bin_inds[0], bin_inds[1], bin_inds[2]] / p_test[bin_inds[0], bin_inds[1], bin_inds[2]])

        return np.array(w)

    def run(self, initial_model_xy):
        x_0 = initial_model_xy[0]
        y_0 = initial_model_xy[1]

        self.centers = [(x_0, y_0)]

        n_iters = self.frames.shape[0] - 1


