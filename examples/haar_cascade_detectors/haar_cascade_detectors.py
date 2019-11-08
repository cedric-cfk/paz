import argparse

from paz.processors import Predict
from paz.processors import ToBoxes2D
from paz.processors import DrawBoxes2D
from paz.processors import ConvertColor
from paz.processors import CastImageToInts
from paz.models import HaarCascadeDetector
from paz.core import SequentialProcessor, VideoPlayer

parser = argparse.ArgumentParser(description='Haarcascade detectors example')
parser.add_argument('-m', '--models', nargs='+', type=str,
                    default=['frontalface_default', 'eye'],
                    help='Model name postfix of openCV xml file')
args = parser.parse_args()

detectors = []
for class_arg, model_name in enumerate(args.models):
    model = HaarCascadeDetector(model_name, class_arg)
    detector = SequentialProcessor()
    preprocessing = [ConvertColor('BGR', 'GRAY')]
    detector.add(Predict(model, 'image', 'boxes', preprocessing))
    detector.add(ToBoxes2D(args.models))
    detectors.append(detector)
pipeline = SequentialProcessor(detectors)
pipeline.add(DrawBoxes2D(args.models))
pipeline.add(CastImageToInts())
VideoPlayer((1280, 960), pipeline).start()