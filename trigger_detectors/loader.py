import os
from ..trigger_detector import TriggerDetectorLoader
from .helper import *

class MyTriggerDetectorLoader(TriggerDetectorLoader):
    
    def __init__(self, default_nli_path=None, agenda_dir=None):
        super(MyTriggerDetectorLoader, self).__init__(\
                default_nli_path=default_nli_path)

        for path in os.listdir(agenda_dir):
            filename = path.split('/')[-1]
            t = filename.split('_')[0]

            if "main" in filename:
                agenda_name = t + "_main"
                print("Loading {} kickoff trigger detector...".format(agenda_name))
                self.register_detector_for_agenda(agenda_name, KickoffTriggerDetector(t))
                print("Loading {} transition trigger detector...".format(agenda_name))
                self.register_detector_for_agenda(agenda_name, BasicTriggerDetector(t))

            elif "push_back" in filename:
                agenda_name = t + "_push_back"
                print("Loading {} pushback trigger detector...".format(agenda_name))
                self.register_detector_for_agenda(agenda_name, PushbackStarterTriggerDetector(t))

