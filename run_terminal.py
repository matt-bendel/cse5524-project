from utils.parse_args import create_arg_parser
from utils import get_bounding_box_coords
from data import load_data
from vision import TemplateMatcher, MeanShiftTracker

if __name__ == '__main__':
    args = create_arg_parser().parse_args()

    # TODO: Load video
    video = load_data(args.data_path)

    frame = args.template_frame
    object = args.template_label

    # TODO: Extract template from first frame and get start coords
    template = None
    template_height = None
    template_width = None
    start_x = None
    start_y = None

    # Get initial model
    template_matcher = TemplateMatcher(template)
    initial_center = template_matcher.run(video[0], start_x, start_y)

    bounding_box_coords = {
        'mean_shift': [],
        'covariance': [],
        'klt': []
    }

    # TODO: Run algos
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
