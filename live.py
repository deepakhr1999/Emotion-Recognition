import dlib
from emotion import EmotionDetector
import utils
import cv2
from PIL import Image
import numpy as np
import pickle
import sys

cap = cv2.VideoCapture(0) 
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("../models/shape_predictor_68_face_landmarks.dat")
# image = np.array(Image.open(sys.argv[1]))
# features = utils.getLandmarks(image, detector, predictor)

model = EmotionDetector.load('../models/emotion.pkl')

while True:  
  
    # reads frames from a camera 
    ret, img = cap.read()  
    s = img.shape[:2]
    s = int(1.5*s[1]), int(1.5*s[0])
    img = Image.fromarray(img).resize(s, Image.ANTIALIAS)
    img = np.array(img)
    # convert to gray scale of each frames 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
  
    res, features, points = utils.getLandmarks(img, detector, predictor)
    if res:
        classname = model.predictClass(features)
        print(classname)
        for point in points:
            cv2.circle(img, point, 3, (255,0,0), 1)
        cv2.putText(img, classname, s, cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0))
    else:
        print("no face!")
    # Display an image in a window 
    cv2.imshow('img',img) 
  
    # Wait for Esc key to stop 
    k = cv2.waitKey(30) & 0xff
    if k == 27: 
        break