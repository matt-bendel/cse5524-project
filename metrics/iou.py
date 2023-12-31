def calculate_iou(gt_bbox, pred_bbox):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes in the same frame.

    Parameters
    ----------
    gt_bbox : [[bl, br], [tl, tr]]
    pred_bbox : [[bl, br], [tl, tr]]

    Returns
    -------
    float
        in [0, 1]
    """
    gt_x1 = gt_bbox[0][0][0]
    gt_y1 = gt_bbox[0][0][1]
    gt_x2 = gt_bbox[0][1][0]
    gt_y2 = gt_bbox[1][0][1]

    pred_x1 = pred_bbox[0][0][0]
    pred_y1 = pred_bbox[0][0][1]
    pred_x2 = pred_bbox[0][1][0]
    pred_y2 = pred_bbox[1][0][1]

    inter_x1 = max(gt_x1, pred_x1)
    inter_y1 = max(gt_y1, pred_y1)
    inter_x2 = min(gt_x2, pred_x2)
    inter_y2 = min(gt_y2, pred_y2)

    inter_area = max(0, inter_x2 - inter_x1 + 1) * max(0, inter_y2 - inter_y1 + 1)
    gt_area = (gt_x2 - gt_x1 + 1) * (gt_y2 - gt_y1 + 1)
    pred_area = (pred_x2 - pred_x1 + 1) * (pred_y2 - pred_y1 + 1)
    union_area = gt_area + pred_area - inter_area

    iou = inter_area / union_area if union_area > 0 else 0

    return iou
