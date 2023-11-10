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
    start_x = template_width // 2
    start_y = template_height // 2

    exit()

    # TODO: Get initial model
    template_matcher = TemplateMatcher(template)
    initial_center = template_matcher.run(video[0], start_x, start_y)

    exit()

    bounding_box_coords = {
        'mean_shift': [],
        'covariance': [],
        'klt': []
    }

    exit()

    if args.mean_shift:
        mean_shift = MeanShiftTracker(video, 16, 25, eps=1e-2)
        centers = mean_shift.run(initial_center)  # Returns the center of the tracked object for each frame

        for center in centers:
            bounding_box_coords['mean_shift'].append(get_bounding_box_coords(center, template_height, template_width))

    if args.covarince:
        pass  # TODO

    if args.klt:
        pass  # TODO

    # TODO: Save results, i.e. reconstruct video
