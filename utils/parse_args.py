import argparse
import pathlib

class Args(argparse.ArgumentParser):
    """
    Defines global default arguments.
    """

    def __init__(self, **overrides):
        """
        Args:
            **overrides (dict, optional): Keyword arguments used to override default argument values
        """

        super().__init__(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        # Override defaults with passed overrides
        self.set_defaults(**overrides)

def create_arg_parser():
    # CREATE THE PARSER
    parser = Args()

    parser.add_argument('--mean-shift', action='store_true',
                        help='Whether or not to run mean shift.')
    parser.add_argument('--covariance', action='store_true',
                       help='Whether or not to run covariance tracking.')
    parser.add_argument('--klt', action='store_true',
                       help='Whether or not to run klt.')
    parser.add_argument('--debug', action='store_true',
                        help='Whether or not to print/display debug info.')

    parser.add_argument('--template-frame', type=int,
                       help='The frame from which the template is taken.', required=True)
    parser.add_argument('--template-label', type=int,
                       help='The template label index.', required=True)
    parser.add_argument('--debug-frames', type=int,
                        help='In debug mode how many frames to use for motion tracking.', default=10)

    parser.add_argument('--data-path', type=pathlib.Path,
                        help='Path to the test video directory.')
    parser.add_argument('--data-name', type=str,
                        help='Folder containing the data.')
    parser.add_argument('--out-dir', type=pathlib.Path,
                       help='Path for outputs (if any) to be stored.')

    return parser