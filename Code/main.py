import os
import numpy as np
import Config
from tensorflow.keras.models import load_model

from Real_time import rt
from Load_images import load_cv2
from Train_model import get_final_model

# Create dictionary of key & value for labels
classNames = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E',
              5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
              10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O',
              15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T',
              20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y',
              25: 'Z', 26: 'del', 27: 'nothing', 28: 'space'}

# Load based models and ensemble model's weights
model1 = load_model(Config.vgg19Path)
model2 = load_model(Config.mobilenetPath)
model3 = load_model(Config.resnet50Path)
modelWeightPath = Config.enDLW

# Get ensemble model
model = get_final_model(model1, model2, model3, modelWeightPath, modeltype='DL')
# Open OpenCV frame for real-time classification
rt(model, classNames)


