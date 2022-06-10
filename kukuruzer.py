from skimage import transform
from skimage import filters
import cv2 as cv
import numpy as np


CASCADE_CLASSIFIER_PATH = 'haarcascade_frontalface_default.xml'


def detect_faces(img: np.ndarray, bounding_box_addition: int):
    face_cascade = cv.CascadeClassifier(CASCADE_CLASSIFIER_PATH)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    face_crops = []
    for (x, y, w, h) in faces:
        min_y = max(y - bounding_box_addition, 0)
        max_y = min(y + h + bounding_box_addition, img.shape[0])
        min_x = max(x - bounding_box_addition, 0)
        max_x = min(x + w + bounding_box_addition, img.shape[1])
        face_crops.append(img[min_y:max_y, min_x:max_x])
    return face_crops


def seam_carving(img: np.ndarray, horizontal_seams: int = 100, vertical_seams: int = 100):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    sobel = filters.sobel(gray.astype("float"))
    carved = transform.seam_carve(img, sobel, 'vertical', vertical_seams)
    carved = np.uint8(carved * 255)
    carved_gray = cv.cvtColor(carved, cv.COLOR_BGR2GRAY)
    carved_sobel = filters.sobel(carved_gray.astype("float"))
    carved = transform.seam_carve(carved, carved_sobel, 'horizontal', horizontal_seams)
    return np.uint8(carved * 255)


def distort_image(path_to_image: str):
    img = cv.imread(path_to_image)
    faces = detect_faces(img, 50)
    for face in faces:
        distorted_face = seam_carving(face)
        cv.imshow('Face', distorted_face)
        cv.waitKey(0)
