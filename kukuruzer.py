from skimage import transform
from skimage import filters
import cv2 as cv
import numpy as np
from abc import ABC, abstractmethod


class FaceDetector(ABC):

    @abstractmethod
    def detect_faces(self, img: np.ndarray) -> list:
        pass


class FaceDistortionProcessor(ABC):

    @abstractmethod
    def distort_face(self, img: np.ndarray) -> np.ndarray:
        pass


class CascadeClassifierFaceDetector(FaceDetector):

    def __init__(self, config_path: str):
        self.instance = cv.CascadeClassifier(config_path)
        self.scale_factor = 1.1
        self.min_neighbors = 4

    def detect_faces(self, img: np.ndarray) -> list:
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        faces_bounding_boxes = self.instance.detectMultiScale(gray, self.scale_factor, self.min_neighbors)
        return [(x, y, x + w, y + h) for x, y, w, h in faces_bounding_boxes]


class SkimageSeamCarvingDistortionProcessor(FaceDistortionProcessor):

    def distort_face(self, img: np.ndarray) -> np.ndarray:
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        sobel = filters.sobel(gray.astype("float"))
        carved = transform.seam_carve(img, sobel, 'vertical', 120)
        carved = np.uint8(carved * 255)
        carved_gray = cv.cvtColor(carved, cv.COLOR_BGR2GRAY)
        carved_sobel = filters.sobel(carved_gray.astype("float"))
        carved = transform.seam_carve(carved, carved_sobel, 'horizontal', 120)
        return np.uint8(carved * 255)


def get_images_by_bounding_boxes(img: np.ndarray, bounding_boxes: list, addition: int = 50):
    h, w, _ = img.shape

    cropped_images = []
    for min_x, min_y, max_x, max_y in bounding_boxes:
        min_x = max(min_x - addition, 0)
        min_y = max(min_y - addition, 0)
        max_x = min(max_x + addition, img.shape[1])
        max_y = min(max_y + addition, img.shape[0])
        cropped_image = img[min_y:max_y, min_x:max_x]
        cropped_images.append(cropped_image)

    return cropped_images


class DistortionContext:

    def __init__(self, face_detector: FaceDetector, face_distortion_processor: FaceDistortionProcessor):
        self._face_detector = face_detector
        self._face_distortion_processor = face_distortion_processor

    @property
    def face_detector(self) -> FaceDetector:
        return self._face_detector

    @face_detector.setter
    def face_detector(self, detector: FaceDetector) -> None:
        self._face_detector = detector

    @property
    def face_distortion_processor(self) -> FaceDistortionProcessor:
        return self._face_distortion_processor

    @face_distortion_processor.setter
    def face_distortion_processor(self, distortion_processor: FaceDistortionProcessor) -> None:
        self._face_distortion_processor = distortion_processor

    def distort_image(self, path_to_image: str):
        img = cv.imread(path_to_image)
        bounding_boxes = self._face_detector.detect_faces(img)
        cropped_face_images = get_images_by_bounding_boxes(img, bounding_boxes)

        filetype = path_to_image.split('.')[-1]
        filename = path_to_image[:-len(filetype)-1]

        file_paths = []
        for i in range(len(cropped_face_images)):
            cropped_face_image = cropped_face_images[i]
            distorted_face_image = self._face_distortion_processor.distort_face(cropped_face_image)
            new_file_path = f'{filename}_{i}.{filetype}'
            cv.imwrite(new_file_path, distorted_face_image)
            file_paths.append(new_file_path)
        return file_paths
