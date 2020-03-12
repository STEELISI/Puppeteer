from os.path import dirname, join, realpath

from nlu import SpacyEngine
from trigger_detector import TriggerDetectorLoader
from trigger_detectors.intent import MessageIntentTriggerDetector
from trigger_detectors.location import (
    CityInExtractionsTriggerDetector,
    LocationInMessageTriggerDetector
)


class MyTriggerDetectorLoader(TriggerDetectorLoader):
    
    def __init__(self, default_snips_path=None):
        super(MyTriggerDetectorLoader, self).__init__(default_snips_path=default_snips_path)
        
        # Our custom trigger detectors.
        
        # Used by make_payment
        self.register_detector(MessageIntentTriggerDetector("payment", "payment"))

        # Used by get_location
        nlp = SpacyEngine.load()
        rootdir = dirname(realpath(__file__))
        snips_paths = [join(rootdir, "../../turducken/data/training/puppeteer/get_location/i_live")]
        cities_path = join(rootdir, "../../turducken/data/dictionaries/cities.txt")
        self.register_detector(CityInExtractionsTriggerDetector())
        self.register_detector(LocationInMessageTriggerDetector(snips_paths, cities_path, nlp))
