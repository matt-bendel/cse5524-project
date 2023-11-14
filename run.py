import os

from ui import UIHandler

base_dir = os.path.dirname(__file__)

uiHandler = UIHandler(base_dir)
uiHandler.run()
