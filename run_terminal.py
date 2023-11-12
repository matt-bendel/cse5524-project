from utils.parse_args import create_arg_parser
from utils import get_bounding_box_coords
from data import FrameData
from vision import TemplateMatcher, MeanShiftTracker

import matplotlib.pyplot as plt

if __name__ == '__main__':
    args = create_arg_parser().parse_args()

    # Load video and extract target bbox
    video_data = FrameData(args.data_path, args.data_name)
    template_bbox = video_data.get_target_bbox_info(args.template_frame, args.template_label)

    # Extract template and pertinent info
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

    if args.debug:
        plt.imshow(template.astype('uint8'))
        plt.savefig('out/true_template.png')

        coords = get_bounding_box_coords(initial_center, template_height, template_width)
        temp = video_data.frames[0, coords[0][0][1]:coords[1][0][1], coords[0][0][0]:coords[0][1][0], :]
        plt.imshow(temp.astype('uint8'))
        plt.savefig('out/extracted_template.png')

    bounding_box_coords = {
        'mean_shift': [],
        'covariance': [],
        'klt': []
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
        pass  # TODO

    if args.klt:
        pass  # TODO

    # TODO: Save results, i.e. reconstruct video
