import cv2 as cv
import numpy as np


CASCADE_CLASSIFIER_PATH = 'haarcascade_frontalface_default.xml'


def detect_faces(img: np.ndarray):
    face_cascade = cv.CascadeClassifier(CASCADE_CLASSIFIER_PATH)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    face_crops = []
    for (x, y, w, h) in faces:
        face_crops.append(img[y:y+h, x:x+w])
    return face_crops


def distort_image(path_to_image: str, scale: float = 0.6):
    img = cv.imread(path_to_image)
    faces = detect_faces(img)
    for face in faces:
        cv.imshow('Face', face)
        cv.waitKey(0)
