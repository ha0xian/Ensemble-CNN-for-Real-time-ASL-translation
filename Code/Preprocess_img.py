import os
import Config
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model


def draw(hand_landmarks,w,h,frame,rec=True):
    """
    Calculate the maximum and minimum value of the hand landmarks to form a rectangle around the hand
    :param hand_landmarks: hand landmarks detected by MediaPipe
    :param w: width of the frame
    :param h: height of the frame
    :param frame: input frame
    :param rec: Boolean to draw a rectangle on the frame
    :return: 4 co-ordinations of the rectangle around the hands
    """
    for handLMs in hand_landmarks:
        x_max = 0
        y_max = 0
        x_min = w
        y_min = h
        for lm in handLMs.landmark:
            x, y = int((lm.x * w)), int((lm.y * h))
            if x > x_max:
                x_max = x
            if x < x_min:
                x_min = x
            if y > y_max:
                y_max = y
            if y < y_min:
                y_min = y
    if rec:
        cv2.rectangle(frame, (x_min-20, y_min-20), (x_max+20, y_max), (0, 255, 0), 2)
    return max(0,x_min-20), max(0,y_min-20), min(x_max+20,w), min(y_max+20,h)

def preprocess_img(img, imgW=256, imgH=256):
    """
    Function to pre-process when images are loaded
    :param img: input image
    :param imgW: image width
    :param imgH: image height
    :return:
    """
    img = cv2.imread(str(img), cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (imgW, imgH), interpolation=cv2.INTER_AREA)
    img = np.array(img)
    img = img.astype('float32')
#     img /= 255
    return img

def preprocess_4model(img, imgW=128,imgH=128,color=True):
    img = cv2.resize(img, (imgW, imgH))
    x = np.array(img, dtype=float)
    x = x.reshape(1, *x.shape)

    return x

def crop_hand(img):
    """
    Detect hand landmarks from input imaeg with MediaPipe, get co-ordinations from draw(), and cropped the image
    :param img: input image
    :return: cropped image
    """
    detected = False
    mp_hands = mp.solutions.hands
    with mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.5) as hands:
        image = cv2.flip(img,1)
        h,w,_=image.shape
        image= cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        frame = image
        results = hands.process(image)

        if results.multi_hand_landmarks:
            x_min, y_min, x_max, y_max = draw(results.multi_hand_landmarks,w,h,image,False)
            detected=True
            frame= image[y_min:y_max, x_min:x_max].copy()
        return frame, detected

def save_crop_hands(inputPath, outputPath='../Cropped_hand'):
    # Save crop hands with crop_hand() into given directory
    detectedNum, totalNum = 0,0
    newFile = Path(outputPath)
    newFile.mkdir(parents=True, exist_ok=True)
    for dir1 in os.listdir(inputPath):
        print(str(dir1), 'is Processing')
        newSubFile=Path(os.path.join(newFile,dir1))
        newSubFile.mkdir(parents=True, exist_ok=True)
        for i in os.listdir(os.path.join(inputPath, dir1)):
            d=False
            totalNum+=1
            imgPath = os.path.join(inputPath, dir1, i)
            image = cv2.imread(imgPath, cv2.COLOR_BGR2RGB)
            croppedImage, detected= crop_hand(image)
            if detected:
                detectedNum+=1
            cv2.imwrite(os.path.join(newFile,newSubFile,i),cv2.cvtColor(croppedImage, cv2.COLOR_BGR2RGB))
    print('Total Image of: {0}, Number of Hand cropped: {1}'.format(totalNum, detectedNum))

def canny_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    ret, _ = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    outImg = cv2.Canny(img, 0.3 * ret, ret)
    return outImg

def threshold_image(img):
    kernel = np.ones((2, 2), np.uint16)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    closing = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=1)
    sure_bg = cv2.dilate(closing, kernel, iterations=3)
    dist_transform = cv2.distanceTransform(sure_bg, cv2.DIST_L2, 3)
    ret, sure_fg = cv2.threshold(dist_transform, 0.1 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    return sure_fg

def adaptive_canny(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(img, (7, 7), 0)
    th = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 13, 4)
    ret, _ = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    outputImg = cv2.Canny(th, ret * 0.5, ret)

    return outputImg

def auto_canny(img, sigma=0.5):
    kernel = np.ones((3,3), np.uint16)
    v = np.median(img)
    lower = int(max(0,(1.0-sigma)*v))
    upper= int(min(255,(1.0+sigma)*v))
    outputImg = cv2.GaussianBlur(img, (7,7), 0)
    outputImg = cv2.Canny(img, lower, upper)
    outputImg = cv2.dilate(outputImg, kernel, iterations=2)
    outputImg = cv2.erode(outputImg,kernel, iterations=1)
    return outputImg

def pred_img(img, model):
    """
    Preprocess image to fit into encoder-decoder network,
    and predict the image with given model to generate a mask
    :param img: input image for predictions
    :param model: encoder-decoder network
    :return: gray-scale output image
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.array(img)
    img = img.astype('float32')/255
    img = cv2.resize(img, (128,128))
    imgM = np.expand_dims(img, axis=0)
    preds = -1*model.predict(imgM)+1
    p = preds
    p = p.reshape((128,128))
    return p*255

def mask_segment(img, mask, size=(128, 128), inverse=False):
    """
    Function take in mask image, pre-process mask image, and apply in on RGB input image
    :param img: RGB of input image
    :param mask: Gray-scale of mask image
    :param size: specifying size to rescale image to
    :param inverse: Boolean to inverse gray-scale mask images
    :return: A processed image
    """
    mask = cv2.resize(mask, size)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    kernelNum = 5
    kernel = np.ones((kernelNum, kernelNum))
    _, mask = cv2.threshold(mask, 235, 255, cv2.THRESH_BINARY)
    if inverse:
        mask = cv2.bitwise_not(mask)
    dilate = cv2.erode(mask, kernel, 5)
    blackPix = np.sum(mask == 0)
    whitePix = np.sum(mask == 255)
    img = cv2.resize(img, size)

    if whitePix >= (blackPix // 4):
        output = cv2.bitwise_and(img, img, mask=dilate)
    else:
        output = img.copy()
    return output


def generate_mask(inputPath, outputPath, model=None):
    """
    Load images from inputPath, and predict image with given model with pred_img(), output image to outputPath
    :param inputPath: directory of input images
    :param outputPath: directory of output images
    :param model: encoder-decoder model
    :return: a grayscale image save in directory outputPath
    """
    if model == None:
        return print('No model given')

    Path(outputPath).mkdir(parents=True, exist_ok=True)

    for dir1 in os.listdir(inputPath):
        dir1Path = os.path.join(inputPath, dir1)
        Path(os.path.join(outputPath, dir1)).mkdir(parents=True, exist_ok=True)
        print('{} is processing'.format(dir1))
        imgList = os.listdir(dir1Path)

        for file in range(len(os.listdir(dir1Path))):
            img = cv2.imread(os.path.join(dir1Path, imgList[file]))
            img = pred_img(img, model)
            cv2.imwrite(os.path.join(outputPath, dir1, imgList[file]), img)
    print('DONE')

def apply_mask(inputPath, outputPath, maskPath,inverse=False):
    """
    Pre-process mask images, and apply it to input images, and output it to outputPath
    :param inputPath: directory of image
    :param outputPath: directory of output images
    :param maskPath: directory of mask images
    :param inverse: Boolean to inverse gray-scale mask images
    :return: save the processed images into outputPath directory
    """
    Path(outputPath).mkdir(parents=True, exist_ok=True)

    for dir1 in os.listdir(inputPath):
        maskdir1Path = os.path.join(maskPath, dir1)
        dir1Path = os.path.join(inputPath, dir1)
        Path(os.path.join(outputPath, dir1)).mkdir(parents=True, exist_ok=True)
        print('{} is processing'.format(dir1))
        imgList = os.listdir(dir1Path)

        for file in range(len(os.listdir(dir1Path))):
            img = cv2.imread(os.path.join(dir1Path, imgList[file]))
            imgString = imgList[file].split('.')[0]
            maskList = os.listdir(maskdir1Path)
            maskString = maskList[file].split('.')[0]
            # Check if image and mask are the same image
            if maskString == imgString:
                mask = cv2.imread(os.path.join(maskdir1Path, maskList[file]))
                img = mask_segment(img, mask, size=(128, 128), inverse=inverse)
                cv2.imwrite(os.path.join(outputPath, dir1, imgList[file]), img)
            else:
                print(maskString, imgString, 'is different')
    print('DONE')

def segment_img(img, model, inverse=False):
    mask = pred_img(img,model)
    output = mask_segment(img, mask,size=(128,128),inverse=inverse)
    return output


if __name__ == "__main__":

    traininputPath = Config.trainDataPath
    traincropPath = Config.cropTrainPath
    trainmaskPath = Config.maskTrainPath
    trainoutputPath = Config.segmentTrainPath

    testinputPath1 = Config.testDataPath1
    testcropPath1 = Config.cropTestPath1
    testmaskPath1 = Config.maskTestPath1
    testsegmentPath1 = Config.segmentTestPath1

    testinputPath2 = Config.testDataPath2
    testcropPath2 = Config.cropTestPath2
    testmaskPath2 = Config.maskTestPath2
    testsegmentPath2 = Config.segmentTestPath2

    unet = load_model(Config.UNetPath)

    save_crop_hands(traininputPath,traincropPath)
    save_crop_hands(testinputPath1,testcropPath1)
    save_crop_hands(testinputPath2,testcropPath2)

    generate_mask(traincropPath, trainmaskPath, 5, model=unet)
    apply_mask(traincropPath, trainoutputPath,6,maskPath=trainmaskPath)

    generate_mask(testcropPath1, testmaskPath1, 5, model=unet)
    apply_mask(testcropPath1, testsegmentPath1,6,maskPath=testmaskPath1)

    generate_mask(testcropPath2, testmaskPath2, 5, model=unet)
    apply_mask(testcropPath2, testsegmentPath2,6,maskPath=testmaskPath2)