import numpy as np
from Preprocess_img import preprocess_img, preprocess_4model
from Evaluation import get_img_text

def predict_image(img, labelMap,model):
    """
    predict the input image to respective label with given model
    :param img: input images
    :param labelMap: Dictionary to map between label and int values
    :param model: classdication model
    :return: prediction probabilities, and image's class
    """
    img = preprocess_img(img)
    prediction = model.predict(np.expand_dims(img, axis=0))[0]
    imgClass = get_img_text(prediction, labelMap)
    print("This image belong in class: ", imgClass, "with ", np.max(prediction))
    return prediction, imgClass
