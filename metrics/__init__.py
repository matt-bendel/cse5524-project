from .accuracy import calculate_accuracy
from .ssim import calculate_ssim
import tqdm
import numpy as np

def eval(video_data, template_label, pred_bboxes):
    accuracy = []
    ssim = []
    frames = video_data.frames
    n_iters = frames.shape[0] - 1

    for i in tqdm(range(n_iters)):
        frame = frames[i]
        video_data.get_target(frame, template_label)
        accuracy.append(calculate_accuracy)
        ssim.append(calculate_ssim)

    return np.mean(accuracy), np.mean(ssim)

