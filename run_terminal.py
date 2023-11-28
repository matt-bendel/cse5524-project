import json
from utils.parse_args import create_arg_parser
from utils import get_bounding_box_coords
from utils.reconstruct_video import reconstruct_video
from data import FrameData
from vision import TemplateMatcher, MeanShiftTracker, CovarianceTracker, KLTTracker
from metrics import eval

import ast
import matplotlib.pyplot as plt
from datetime import datetime

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

    template_matcher = TemplateMatcher(template)
    initial_center = template_matcher.run(video_data.frames[0], start_x, start_y, end_x=end_x, end_y=end_y)
    print(initial_center)
    # initial_center = (518, 604) # debug 1

    if args.debug:
        plt.imshow(template.astype('uint8'))
        plt.savefig('out/true_template.png')

        coords = get_bounding_box_coords(initial_center, template_height, template_width)
        temp = video_data.frames[0, coords[0][0][1]:coords[1][0][1], coords[0][0][0]:coords[0][1][0], :]
        plt.imshow(temp.astype('uint8'))
        plt.savefig('out/extracted_template.png')

    gt_bbox = video_data.get_all_bbox_info(args.template_label, 300)
    current_datetime = datetime.now().strftime("%Y-%m-%d %H-%M-%S")

    # output_path = f'out/gt_video_{args.template_label}.mp4' # test 
    # reconstruct_video(output_path, video_data.frames[:300, :, :, :], gt_bbox)
    
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
            # if args.debug:
            #     temp = video_data.frames[count, coords[0][0][1]:coords[1][0][1], coords[0][0][0]:coords[0][1][0], :]
            #     plt.imshow(temp.astype('uint8'))
            #     plt.savefig(f'out/ms_{count}.png')

            # count += 1

            bounding_box_coords['mean_shift'].append(coords)

        output_path = 'out/mean_shift_video' + str(current_datetime) + '.mp4'
        reconstruct_video(output_path, video, bounding_box_coords['mean_shift'])
        iou, ssim = eval(video, bounding_box_coords['mean_shift'], gt_bbox)
        accuracy['mean_shift']['iou'] = iou
        accuracy['mean_shift']['ssim'] = ssim   

    if args.covariance:
        initial_bbox = get_bounding_box_coords(initial_center, template_height, template_width)
        video = video_data.frames

        if args.debug and args.debug_frames > 0:
            video = video[0:args.debug_frames]
        
        covariance = CovarianceTracker(video, template_height, template_width)
        bboxs = covariance.run(initial_bbox)

        # count = 0
        # for bbox in bboxs:
        #     if args.debug:
        #         temp = video_data.frames[count, bbox[0][0][1]:bbox[1][0][1], bbox[0][0][0]:bbox[0][1][0], :]
        #         plt.imshow(temp.astype('uint8'))
        #         plt.savefig(f'out/ms_{count}.png')

        #     count += 1

        bounding_box_coords['covariance'] = bboxs
        output_path = 'out/covariance_video' + str(current_datetime) + '.mp4'
        reconstruct_video(output_path, video, bounding_box_coords['covariance'])
        iou, ssim = eval(video, bounding_box_coords['covariance'], gt_bbox)
        accuracy['covariance']['iou'] = iou
        accuracy['covariance']['ssim'] = ssim   

    if args.klt:
        initial_bbox = get_bounding_box_coords(initial_center, template_height, template_width)
        video = video_data.frames

        if args.debug and args.debug_frames > 0:
            video = video[0:args.debug_frames]
        
        klt = KLTTracker(video)
        bboxs = klt.run(initial_bbox)

        # count = 0
        # for bbox in bboxs:
        #     if args.debug:
        #         temp = video_data.frames[count, bbox[0][0][1]:bbox[1][0][1], bbox[0][0][0]:bbox[0][1][0], :]
        #         plt.imshow(temp.astype('uint8'))
        #         plt.savefig(f'out/ms_{count}.png')

        #     count += 1

        bounding_box_coords['klt'] = bboxs
        bounding_box_coords['klt'] = bboxs
        output_path = 'out/klt_video' + str(current_datetime) + '.mp4'
        reconstruct_video(output_path, video, bounding_box_coords['klt'])
        iou, ssim = eval(video, bounding_box_coords['klt'], gt_bbox)
        accuracy['klt']['iou'] = iou
        accuracy['klt']['ssim'] = ssim 
        
    print(accuracy)
