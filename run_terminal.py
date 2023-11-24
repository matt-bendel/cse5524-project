import json
from utils.parse_args import create_arg_parser
from utils import get_bounding_box_coords
from utils.reconstruct_video import reconstruct_video
from data import FrameData
from vision import TemplateMatcher, MeanShiftTracker, CovarianceTracker
from metrics import eval
import ast

import matplotlib.pyplot as plt

if __name__ == '__main__':
    args = create_arg_parser().parse_args()

    # Load video and extract target bbox
    video_data = FrameData(args.data_path, args.data_name)
    template_bbox = video_data.get_target_bbox_info(args.template_frame, args.template_label)

    template = video_data.extract_initial_template(args.template_frame, template_bbox)
    template_height = template.shape[0]
    template_width = template.shape[1]

    # Cheat a little here, so it doesn't take an eternity to run...
    # Key Assumption: Not a lot of motion across seconds
    # start_y = template_height // 2 # Correct start, but too slow...
    start_y = template_bbox['top'] - template_height // 4
    if start_y < template_height // 2:
        start_y = template_height // 2

    end_y = start_y + template_height + template_height // 4

    # start_x = template_width // 2 # Correct start, but too slow...
    start_x = template_bbox['left'] - template_width // 4
    if start_x < template_width // 2:
        start_x = template_width // 2

    end_x = start_x + template_width + template_width // 4

    # template_matcher = TemplateMatcher(template)
    # initial_center = template_matcher.run(video_data.frames[0], start_x, start_y, end_x=end_x, end_y=end_y)
    initial_center = (519, 534) # Debug

    # if args.debug:
    #     plt.imshow(template.astype('uint8'))
    #     plt.savefig('out/true_template.png')

    #     coords = get_bounding_box_coords(initial_center, template_height, template_width)
    #     temp = video_data.frames[0, coords[0][0][1]:coords[1][0][1], coords[0][0][0]:coords[0][1][0], :]
    #     plt.imshow(temp.astype('uint8'))
    #     plt.savefig('out/extracted_template.png')

    gt_bbox = video_data.get_all_bbox_info(args.template_label, args.debug_frames)

    output_path = f'out/gt_video.mp4' # test
    # reconstruct_video(output_path, video, gt_bbox)
        
    bounding_box_coords = {
        'mean_shift': [],
        'covariance': [],
        'klt': []
    }

    accuracy = {
        'mean_shift': {
            'iou': None,
            'ssim': None
        },
        'covariance': {
            'iou': None,
            'ssim': None
        },
        'klt': {
            'iou': None,
            'ssim': None
        }
    }

    if args.mean_shift:
        video = video_data.frames

        if args.debug and args.debug_frames > 0:
            video = video[0:args.debug_frames]

        mean_shift = MeanShiftTracker(video, 16, template_width // 2)
        # Due to radius of template width, shift up y coord of center to better track person
        centers = mean_shift.run(initial_center)  # Returns the center of the tracked object for each frame

        count = 0
        for center in centers:
            coords = get_bounding_box_coords(center, template_height, template_width)
            if args.debug:
                temp = video_data.frames[count, coords[0][0][1]:coords[1][0][1], coords[0][0][0]:coords[0][1][0], :]
                plt.imshow(temp.astype('uint8'))
                plt.savefig(f'out/ms_{count}.png')

            count += 1

            bounding_box_coords['mean_shift'].append(coords)

    if args.covariance:
        initial_bbox = get_bounding_box_coords(initial_center, template_height, template_width)
        video = video_data.frames

        if args.debug and args.debug_frames > 0:
            video = video[0:args.debug_frames]
        
        covariance = CovarianceTracker(video, template_height, template_width)
        bboxs = covariance.run(initial_bbox)

        count = 0
        for bbox in bboxs:
            if args.debug:
                temp = video_data.frames[count, bbox[0][0][1]:bbox[1][0][1], bbox[0][0][0]:bbox[0][1][0], :]
                plt.imshow(temp.astype('uint8'))
                plt.savefig(f'out/ms_{count}.png')

            count += 1

            bounding_box_coords['covariance'].append(bboxs)

    if args.klt:
        pass  # TODO

    # TODO: Save results, i.e. reconstruct video
    output_path = 'out/mean_shift_video.mp4' # test
    file_path = 'out/bbox_coords_result.json'

    # with open(file_path, 'w') as file:
    #     file.write(str(bounding_box_coords))
    # reconstruct_video(output_path, video, bounding_box_coords['mean_shift'])

    with open(file_path, 'r') as file:
        pred_bboxes = file.read()
        pred_bboxes = ast.literal_eval(pred_bboxes)
    
    video = video_data.frames
    iou, ssim = eval(video, pred_bboxes['mean_shift'], gt_bbox)
    accuracy['mean_shift']['iou'] = iou
    accuracy['mean_shift']['ssim'] = ssim

    print(accuracy)
