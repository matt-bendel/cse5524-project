from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import cv2
import numpy as np

def calculate_ssim(frame, gt_bbox, pred_bbox):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes in the same frame.

    Parameters
    ----------
    gt_bbox : [[bl, br], [tl, tr]]
    pred_bbox : [[bl, br], [tl, tr]]

    Returns
    -------
    float
        in [-1, 1]
    """
    gt_x1 = gt_bbox[0][0][0]
    gt_y1 = gt_bbox[0][0][1]
    gt_x2 = gt_bbox[0][1][0]
    gt_y2 = gt_bbox[1][0][1]

    pred_x1 = pred_bbox[0][0][0]
    pred_y1 = pred_bbox[0][0][1]
    pred_x2 = pred_bbox[0][1][0]
    pred_y2 = pred_bbox[1][0][1]

    frame = frame.astype(np.uint8)
    gt_roi = frame[gt_y1:gt_y2, gt_x1:gt_x2, :]
    pred_roi = frame[pred_y1:pred_y2, pred_x1:pred_x2, :]

    gt_roi = cv2.cvtColor(gt_roi, cv2.COLOR_BGR2GRAY) if len(gt_roi.shape) > 2 else gt_roi
    pred_roi = cv2.cvtColor(pred_roi, cv2.COLOR_BGR2GRAY) if len(pred_roi.shape) > 2 else pred_roi

    if gt_roi.shape != pred_roi.shape:
        gt_roi = cv2.resize(gt_roi, (pred_roi.shape[1], pred_roi.shape[0]))

    ssim_value, _ = ssim(gt_roi, pred_roi, full=True)

    return ssim_value
