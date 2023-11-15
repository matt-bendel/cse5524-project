import os
import sys

sys.path.append(os.path.abspath(os.path.join('..', 'data')))
sys.path.append(os.path.abspath(os.path.join('..', 'vision')))
sys.path.append(os.path.abspath(os.path.join('..', 'utils')))

from vision import MeanShiftTracker, TemplateMatcher # TODO: Other methods
from data import FrameData
from utils import get_bounding_box_coords
from PIL import Image

import PySimpleGUI as sg
import io
import matplotlib.pyplot as plt

class UIHandler:
    def __init__(self, base_dir):
        self.current_layout = 1
        self.base_dir = base_dir
        self.selected_video = ''
        self.selected_template_id = ''
        self.video_data = None

        self.video_options = self._get_video_options()
        self.tracking_options = ['klt', 'covariance', 'ms']
        self.template_data = {}
        for folder in self.video_options:
            self.template_data[folder] = self._get_template_options(folder)

        layout = self._get_layouts(base_dir)
        self.window = sg.Window('Swapping the contents of a window', layout, size=(900, 600), element_justification='c')

    def _get_layouts(self, base_dir):
        image_data = []
        layout1 = [[sg.Text('Please choose a video', key='video-select-header')]]

        for folder in self.video_options:
            image = Image.open(os.path.join(base_dir, f'data/{folder}/img1/000001.jpg'))
            new_width = 200
            new_height = 115
            image = image.resize((new_width, new_height), Image.ANTIALIAS)
            bio = io.BytesIO()
            image.save(bio, format="PNG")
            image_data.append(bio.getvalue())
            layout1.append(
                sg.pin(sg.Column([
                    [sg.Image(data=bio.getvalue())],
                    [sg.Button('Choose', key=folder)]
                ], vertical_alignment='c', element_justification='c', key=f'video-select-{folder}', pad=(0, 0)
                ))
            )

        layout2 = [[sg.Text('Please select the target to track', key='template-select-header', visible=False)]]
        for folder in self.video_options:
            for template in self.template_data[folder]:
                template_index = template.split(folder+'/')[1].split('.')[0]
                image = Image.open(os.path.join(base_dir, template))
                new_height = 150
                new_width = 150
                image = image.resize((new_width, new_height), Image.ANTIALIAS)
                bio = io.BytesIO()
                image.save(bio, format="PNG")
                image_data.append(bio.getvalue())
                layout2.append(
                    sg.pin(sg.Column([
                        [sg.Image(data=bio.getvalue())],
                        [sg.Button('Choose', key=f'template-select-{folder}-{template_index}-btn')]
                    ], vertical_alignment='c', element_justification='c',
                        key=f'template-select-{folder}-{template_index}',
                        visible=False, pad=(0, 0)
                    ))
                )

        layout2.append([sg.Button('Back to Video Select', key='to-video-select', visible=False)])

        layout3 = [[sg.Text('Please select a tracking algorithm', key='tracking-select-header', visible=False)]]
        layout3.append(sg.pin(sg.Column([
                        [sg.Button('KLT Tracking', key='klt')]
        ], vertical_alignment='c', element_justification='c', key='tracking-select-klt', visible=False, pad=(0, 0)))
        )
        layout3.append(sg.pin(sg.Column([
            [sg.Button('Covariance Tracking', key='covariance')]
        ], vertical_alignment='c', element_justification='c', key='tracking-select-covariance', visible=False, pad=(0, 0)))
        )
        layout3.append(sg.pin(sg.Column([
            [sg.Button('Mean Shift Tracking', key='ms')]
        ], vertical_alignment='c', element_justification='c', key='tracking-select-ms', visible=False, pad=(0, 0)))
        )
        layout3.append([sg.Button('Back to Template Select', key='to-template-select', visible=False)])

        layout4 = [[sg.Text('Please wait while we track your target. This will take a few minutes.', key='please-wait', visible=False)]]

        return [layout1, layout2, layout3, layout4]

        # TODO: Tracking algo results - video, metrics (back to start)

    def _get_video_options(self):
        video_dir = os.listdir(os.path.join(self.base_dir, 'data'))
        parsed_video_options = []

        for file in video_dir:
            if file.find('.py') == -1 and file != '__pycache__':
                parsed_video_options.append(file)

        return parsed_video_options

    def _get_template_options(self, folder):
        template_dir = os.listdir(os.path.join(self.base_dir, f'ui/templates/{folder}'))
        parsed_template_options = []

        for file in template_dir:
            parsed_template_options.append(os.path.join(self.base_dir, f'ui/templates/{folder}/{file}'))

        return parsed_template_options

    def _load_video(self):
        return FrameData(os.path.join(self.base_dir, 'data'), self.selected_video)

    def _perform_template_match(self, video_data, template, template_bbox):
        template_height = template.shape[0]
        template_width = template.shape[1]

        start_y = template_bbox['top'] - template_height // 4
        if start_y < template_height // 2:
            start_y = template_height // 2

        end_y = start_y + template_height + template_height // 4

        start_x = template_bbox['left'] - template_width // 4
        if start_x < template_width // 2:
            start_x = template_width // 2

        end_x = start_x + template_width + template_width // 4

        template_matcher = TemplateMatcher(template)
        initial_center = template_matcher.run(video_data.frames[0], start_x, start_y, end_x=end_x, end_y=end_y)

        return initial_center

    def _perform_motion_tracking(self, method, video_data, template_width, template_height, initial_center):
        bounding_box_coords = []

        if method == 'ms':
            video = video_data.frames
            mean_shift = MeanShiftTracker(video, 16, template_width // 2)
            # Due to radius of template width, shift up y coord of center to better track person
            centers = mean_shift.run(initial_center)  # Returns the center of the tracked object for each frame

            count = 0
            for center in centers:
                coords = get_bounding_box_coords(center, template_height, template_width)
                # TODO: REMOVE BELOW
                temp = video_data.frames[count, coords[0][0][1]:coords[1][0][1],
                       coords[0][0][0]:coords[0][1][0], :]
                plt.imshow(temp.astype('uint8'))
                plt.savefig(f'out/ms_ui_{count}.png')

                count += 1
                # TODO: REMOVE ABOVE

                bounding_box_coords.append(coords)

        elif method == 'covariance':
            pass  # TODO - Ana

        elif method == 'klt':
            pass  # TODO - Ana

        return bounding_box_coords

    def _toggle_video_select(self, show):
        for folder in self.video_options:
            self.window[f'video-select-{folder}'].update(visible=show)

        self.window['video-select-header'].update(visible=show)

    def _toggle_template_select(self, show):
        for folder in self.video_options:
            for template in self.template_data[folder]:
                template_index = template.split(folder+'/')[1].split('.')[0]
                self.window[f'template-select-{folder}-{template_index}'].update(visible=show)

        self.window['to-video-select'].update(visible=show)
        self.window['template-select-header'].update(visible=show)

    def _toggle_tracking_select(self, show):
        for method in self.tracking_options:
            self.window[f'tracking-select-{method}'].update(visible=show)

        self.window['to-template-select'].update(visible=show)
        self.window['tracking-select-header'].update(visible=show)

    def _toggle_progress(self, show):
        self.window['please-wait'].update(visible=show)

    def run(self):
        # Core loop in here
        while True:
            event, values = self.window.read()
            print(event, values)
            if event in (None, 'Exit'):
                break
            if event in self.video_options:
                self.selected_video = event
                self._toggle_video_select(False)
                self._toggle_template_select(True)
            elif event == 'to-video-select':
                self.selected_video = ''
                self._toggle_video_select(True)
                self._toggle_template_select(False)
            elif event.find('template-select-') != -1:
                parsed_str = event.split('-')
                folder = parsed_str[2]
                template_id = parsed_str[3]

                self.selected_template_id = template_id
                self._toggle_template_select(False)
                self._toggle_tracking_select(True)
            elif event == 'to-template-select':
                self.selected_template_id = ''
                self._toggle_template_select(True)
                self._toggle_tracking_select(False)
            elif event in self.tracking_options:
                self._toggle_template_select(False)
                self._toggle_progress(True)

                tracking_method = event

                # Load video and extract target bbox
                video_data = self._load_video()
                template_bbox = video_data.get_target_bbox_info(60, self.selected_template_id)
                template = video_data.extract_initial_template(0, template_bbox)

                initial_center = self._perform_template_match(video_data, template, template_bbox)
                bbox_coords = self._perform_motion_tracking(tracking_method, video_data, template.shape[1], template.shape[0], initial_center)

                # TODO: Ana - get video
                # TODO: Matt - display video

                self._toggle_progress(False)
                # TODO: self._toggle_results(True)
                self.selected_video = ''
                self.selected_template_id = ''
                self._toggle_video_select(True) # TODO: remove


        self.window.close()
