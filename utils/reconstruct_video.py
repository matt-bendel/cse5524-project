import cv2
import numpy as np
from matplotlib.colors import is_color_like

def reconstruct_video(args, output_path, frames, bboxes, color='green'):
    
    if frames.shape[0] != len(bboxes):
        raise ValueError("Number of frames and bounding boxes don't match")
    
    # Define color of the bounidng box
    try:
        is_color_like(str(color))
        color = str(color)
    except ValueError:
        color = 'green'

    height, width, _ = frames[0].shape
    frames_copy = [np.copy(frame) for frame in frames]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 25, (width, height), isColor=True)

    for frame, bbox in zip(frames_copy, bboxes):
        (bl, br), (tl, tr) = bbox
        cv2.rectangle(frame, tl, br, str(color), 2)
        frame = frame.astype(np.uint8)
        out.write(frame)

    out.release()
    cv2.destroyAllWindows()
