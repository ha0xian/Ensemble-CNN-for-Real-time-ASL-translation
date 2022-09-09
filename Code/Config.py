"""
Directory Paths to all the data, models, weights and outputs
"""

# Paths to training image and pre-processed images from https://www.kaggle.com/datasets/grassknoted/asl-alphabet
trainDataPath = '../Data/TrainData'
cropTrainPath = '../Data/Crop_Train'
maskTrainPath = '../Data/Mask_Train'
segmentTrainPath = '../Data/Segmented_Train'

# Paths to testing images and pre-processed images from https://www.kaggle.com/datasets/grassknoted/asl-alphabet
testDataPath1 = '../Data/TestData1'
cropTestPath1 = '../Data/Crop_Test1'
maskTestPath1 = '../Data/Mask_Test1'
segmentTestPath1 = '../Data/Segmented_Test1'

# Paths to testing images and pre-processed images from https://www.kaggle.com/datasets/danrasband/asl-alphabet-test
testDataPath2 = '../Data/TestData2'
cropTestPath2 = '../Data/Crop_Test2'
maskTestPath2 = '../Data/Mask_Test2'
segmentTestPath2 = '../Data/Segmented_Test2'

# Paths of dataset for object-detection model from http://sun.aei.polsl.pl/~mkawulok/gestures/
# and http://www.ouhands.oulu.fi/
dataSegment1 = '../Data/Data_Segmentation/DataSegment1'
dataSegment2 = '../Data/Data_Segmentation/DataSegment2'
dataSegment3 = '../Data/Data_Segmentation/DataSegment3'
dataMask1 = '../Data/Data_Segmentation/DataMask1'
dataMask2 = '../Data/Data_Segmentation/DataMask2'
dataMask3 = '../Data/Data_Segmentation/DataMask3'

# Paths to saved classification models
vgg19Path = '../Models/VGG19_01.h5'
resnet50Path = '../Models/ResNet50_01.h5'
mobilenetPath = '../Models/MobileNet_01.h5'
UNetPath = '../Seg_Models/UNET_01.h5'

# Paths to saved ensemble models weights
enAvgW = '../Weights/ensembleAvg_01.h5'
enMaxW = '../Weights/ensembleMax_01.h5'
enDLW = '../Weights/ensembleDL_01.h5'

# Paths to saved images, graphs and csv files of classification results
resultsPath = '../Results/'



