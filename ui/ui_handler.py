from PIL import Image

import PySimpleGUI as sg

import os
import io


class UIHandler:
    def __init__(self, base_dir):
        self.current_layout = 1
        self.selected_video = ''
        self.selected_template_id = ''
        self.video_data = None

        self.video_options = self._get_video_options(base_dir)
        self.tracking_options = ['klt', 'covariance', 'ms']
        self.template_data = {}
        for folder in self.video_options:
            self.template_data[folder] = self._get_template_options(base_dir, folder)

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

        return [layout1, layout2, layout3]

        # TODO: template match/Tracking algo progress ->
        # TODO: Tracking algo results - video, metrics (back to start)

    def _get_video_options(self, base_dir):
        video_dir = os.listdir(os.path.join(base_dir, 'data'))
        parsed_video_options = []

        for file in video_dir:
            if file.find('.py') == -1 and file != '__pycache__':
                parsed_video_options.append(file)

        return parsed_video_options

    def _get_template_options(self, base_dir, folder):
        template_dir = os.listdir(os.path.join(base_dir, f'ui/templates/{folder}'))
        parsed_template_options = []

        for file in template_dir:
            parsed_template_options.append(os.path.join(base_dir, f'ui/templates/{folder}/{file}'))

        return parsed_template_options

    def _perform_template_match(self):
        pass

    def _perform_motion_tracking(self):
        pass

    def _progress_bar(self):
        pass

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
                tracking_method = event
                print(f'SELECTED TRACKER: {tracking_method}')
                # TODO: hide track select, show progress
                # TODO: Dataload
                # TODO: template match
                # TODO: track
                # TODO: Move to results
                pass

        self.window.close()
