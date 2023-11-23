from skimage.metrics import structural_similarity as ssim
import cv2

def calculate_ssim(frame, gt_bbox, pred_bbox):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes in the same frame.

    Parameters
    ----------
    gt_bbox : [[tl, tr], [bl, br]]
    pred_bbox : [[tl, tr], [bl, br]]

    Returns
    -------
    float
        in [-1, 1]
    """
    gt_tl = gt_bbox[0][0][0]
    gt_br = gt_bbox[0][1][1]
    gt_x1, gt_y1 = gt_tl
    gt_x2, gt_y2 = gt_br

    pred_tl = pred_bbox[0][0][0]
    pred_br = pred_bbox[0][1][1]
    pred_x1, pred_y1 = pred_tl
    pred_x2, pred_y2 = pred_br

    gt_roi = frame[gt_y1:gt_y2, gt_x1:gt_x2]
    pred_roi = frame[pred_y1:pred_y2, pred_x1:pred_x2]

    gt_roi = cv2.cvtColor(gt_roi, cv2.COLOR_BGR2GRAY) if len(gt_roi.shape) > 2 else gt_roi
    pred_roi = cv2.cvtColor(pred_roi, cv2.COLOR_BGR2GRAY) if len(pred_roi.shape) > 2 else pred_roi

    ssim_value, _ = ssim(gt_roi, pred_roi, full=True)

    return ssim_value
