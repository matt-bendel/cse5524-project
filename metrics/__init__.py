from .iou import calculate_iou
from .ssim import calculate_ssim
from tqdm import tqdm 
import numpy as np

def eval(frames, pred_bboxes, gt_bboxes):
    iou = []
    ssim = []
    # n_iters = frames.shape[0] - 1
    n_iters = len(pred_bboxes) # deal with the number of bbox here

    for i in tqdm(range(n_iters)):
        iou.append(calculate_iou(gt_bboxes[i], pred_bboxes[i]))
        ssim.append(calculate_ssim(frames[i, :, :, :], gt_bboxes[i], pred_bboxes[i]))

    return np.mean(iou), np.mean(ssim)

