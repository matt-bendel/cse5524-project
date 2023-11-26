import numpy as np
import os
import cv2
import glob

# GT DATA STRUCTURE:
# Each row is data point w/ following format
# FRAME | TARGET ID | BOX LEFT | BOX TOP | BOX WIDTH | BOX HEIGHT | CONFIDENCE | other stuff not relevant for us
class FrameData:
    def __init__(self, data_dir, data_name):
        # Load GT Data
        file_path = f'{data_dir}/{data_name}/gt/gt.txt'
        data = np.genfromtxt(file_path, delimiter=',')
        if data.ndim == 1:  # Because in MOT we have different delimites in result files?!?!?!?!?!?
            data = np.genfromtxt(file_path, delimiter=' ')

        nan_index = np.sum(np.isnan(data), axis=1)
        self.gt_data = data[nan_index == 0]

        # Metadata
        sequences = np.genfromtxt(os.path.join(f'{data_dir}/{data_name}/seqinfo.ini'), dtype='str', skip_header=True)
        seq_info = {}
        for seq in sequences:
            key, value = seq.split('=')
            seq_info[key] = value

        self.im_height = int(seq_info['imHeight'])
        self.im_width = int(seq_info['imWidth'])
        self.im_ext = seq_info['imExt']
        self.frame_rate = int(seq_info['frameRate'])
        self.seq_length = int(seq_info['seqLength'])

        self.frames = np.zeros((self.seq_length, self.im_height, self.im_width, 3))

        for t, img in enumerate(sorted(glob.glob(os.path.join(f'{data_dir}/{data_name}/img1', f"*{self.im_ext}")))):
            if t == self.seq_length:
                break

            im = cv2.imread(img, 1)
            self.frames[t, :, :, :] = im

        self.frames = self.frames.astype('float64')

    def get_target_bbox_info(self, template_frame, template_id):
        gt_frame_data_inds = np.flatnonzero(self.gt_data[:, 0] == template_frame)
        relevant_info = None

        for ind in gt_frame_data_inds:
            if self.gt_data[ind, 1] == int(template_id):
                relevant_info = self.gt_data[ind]

        return {
            'left': int(relevant_info[2]),
            'top': int(relevant_info[3]),
            'width': int(relevant_info[4]),
            'height': int(relevant_info[5])
        }
    
    def get_all_bbox_info(self, template_id, debug_frames=0):
        gt_bbox = []
        video = self.frames
        if debug_frames != 0:
            video = video[0:debug_frames]

        for idx in range(video.shape[0]):
            bbox = self.get_target_bbox_info(idx+1, template_id)
            bl = (bbox['left'], bbox['top'])
            br = (bbox['left'] + bbox['width'], bbox['top'])
            tl = (bbox['left'], bbox['top'] + bbox['height'])
            tr = (bbox['left'] + bbox['width'], bbox['top'] + bbox['height'])
            bbox = [[bl, br], [tl, tr]]
            gt_bbox.append(bbox)
        return gt_bbox

    def extract_initial_template(self, template_frame, bbox_info):
        return self.frames[template_frame, bbox_info['top']:bbox_info['top']+bbox_info['height'], bbox_info['left']:bbox_info['left']+bbox_info['width'], :]

