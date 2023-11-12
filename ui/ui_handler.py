import PySimpleGUI as sg

class UIHandler:
    def __init__(self):
        self.window = sg.Window(title="Hello World", layout=[[]], margins=(100, 50)).read()
