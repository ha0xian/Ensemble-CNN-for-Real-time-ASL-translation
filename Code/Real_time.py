import time

import cv2
import numpy as np
import mediapipe as mp
from Preprocess_img import draw, preprocess_4model, segment_img

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

def rt(model,className):
    """
    Function to use web-cam for real-time classification with given model labels
    :param model: classification model
    :param className: classification labels
    :return: a frame for real-time classification
    """
    cap = cv2.VideoCapture(0) # Initiate laptop's web-cam
    _, frame = cap.read() # Read laptop's web-cam
    pTime = 0 # Variables to calculate FPS
    h, w, c = frame.shape
    with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:

        while cap.isOpened():
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = hands.process(image) # Detects hands with MediaPipe
            image.flags.writeable = True
            hand_landmarks = results.multi_hand_landmarks

            if hand_landmarks:
                x1, y1, x2, y2 = draw(hand_landmarks, w, h, frame) # Get maximum and minimum landmarks to draw a rectangle on the frame
                rec = frame[y1:y2, x1:x2] # Crop the rectangle from the entire frame
                handPred, handClass = predict_frame(rec, className, model) # Passed cropped frame to model for prediction
                if str(handClass) != 'nothing' and np.max(handPred)>=0.3: # Do not print anything if predicted class is nothing
                    # Print predicted class on frame
                    cv2.putText(frame, text=str(handClass),
                                org=(20, 100),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=3, color=(255, 255, 0),
                                thickness=3, lineType=cv2.LINE_AA)

            cTime = time.time()
            fps = 1 / (cTime - pTime) #Calculate FPS
            pTime = cTime
            cv2.putText(frame, 'fps: ' + str(int(fps)), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2) # Print FPS value on screen
            cv2.imshow('Hand tracking', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'): # press Q to exit
                break

def predict_frame(img, labelMap, model):
    img = preprocess_4model(img)
    prediction = model.predict(img)[0]
    imgClass = labelMap[np.argmax(prediction)]
    print("This image belong in class: ", imgClass, "with ", np.max(prediction))
    return prediction, imgClass

