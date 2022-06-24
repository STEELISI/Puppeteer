from os.path import dirname, join, realpath

from ..trigger_detector import TriggerDetectorLoader
from .website import *
from .shipment import *
from .payment import *

class MyTriggerDetectorLoader(TriggerDetectorLoader):
    
    def __init__(self, default_nli_path=None, default_snips_path=None, agenda_names=None):
        super(MyTriggerDetectorLoader, self).__init__(\
                default_nli_path=default_nli_path, default_snips_path=default_snips_path)
       
        agenda_names = set(agenda_names)
        # Our custom trigger detectors.
        
        # Used by get_website
        if "get_website" in agenda_names:
            #self.register_detector(KickOffWebsiteTriggerDetector("website")) #kickoff
            #self.register_detector_for_agenda("get_website", TransitionWebsiteTriggerDetector("url"))
            self.register_detector_for_agenda("get_website", URLWebsiteTriggerDetector())

        # Used by get_shipment
        if "get_shipment" in agenda_names:
            #self.register_detector(KickOffShipmentTriggerDetector()) #kickoff
            #self.register_detector_for_agenda("get_shipment", TransitionShipmentTriggerDetector()) #transition
            pass

        # Used by get_payment
        if "get_payment" in agenda_names:
            #self.register_detector(KickOffPaymentTriggerDetector()) #kickoff
            #self.register_detector_for_agenda("get_payment", TransitionPaymentTriggerDetector()) #transition
            # self.register_detector(PaymentTriggerDetector("payment")) #kickoff
            # self.register_detector_for_agenda("get_payment", EAccountTriggerDetector("e_account"))
            # self.register_detector_for_agenda("get_payment", AskPaymentTriggerDetector("ask_payment_response"))
            # self.register_detector_for_agenda("get_payment", SignupTriggerDetector("signup"))
            self.register_detector_for_agenda("get_payment", AccountPaymentTriggerDetector())
            self.register_detector_for_agenda("get_payment", OtherPaymentTriggerDetector())

        '''
        # Used by get_location
        nlp = SpacyEngine.load()
        rootdir = dirname(realpath(__file__))
        snips_paths = [join(rootdir, "../../training_data/get_location/i_live")]
        cities_path = join(rootdir, "../../dictionaries/cities.txt")
        self.register_detector(CityInExtractionsTriggerDetector())
        self.register_detector(LocationInMessageTriggerDetector(snips_paths, cities_path, nlp))
        '''

        '''
        # Used by get_location
        nlp = SpacyEngine.load()
        rootdir = dirname(realpath(__file__))
        #snips_paths = [join(rootdir, "../../turducken/data/training/puppeteer/get_location/i_live")]
        #cities_path = join(rootdir, "../../turducken/data/dictionaries/cities.txt")
        snips_paths = [join(rootdir, "../../training_data/get_location/i_live")]
        #print('MTGL: snips_paths={}'.format(snips_paths))
        cities_path = join(rootdir, "../../dictionaries/cities.txt")
        #print('MTGL: cities_paths={}'.format(cities_path))
        self.register_detector(CityInExtractionsTriggerDetector())
        self.register_detector(LocationInMessageTriggerDetector(snips_paths, cities_path, nlp))
        '''

