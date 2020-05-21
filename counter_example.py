import datetime
import cv2
import matplotlib.pyplot as plt
import cvlib as cv
from cvlib.object_detection import draw_bbox
import numpy as np

class CarDetector:

	def __init__(self, video):
		self.video = video
		self.capture = cv2.VideoCapture(video)

	def get_centroid(self, bbox):
		return int((bbox[0] + bbox[2])/2), int((bbox[1] + bbox[3])/2)

	def detect_cars(self):

		detected_cars = []
		ret, frame = self.capture.read()

		try:
			bboxes, labels, conf = cv.detect_common_objects(frame)
			print (bboxes)
		except:
			return detected_cars

		for bbox, label in zip(bboxes, labels):
			if label == 'car':
				detected_cars.append(bbox)

		return detected_cars

	def get_car_centroids(self):
		return [self.get_centroid(bbox) for bbox in self.detect_cars()]

class Counter:

    def __init__(self, place, direction, counter_left_x, counter_left_y, counter_right_x, counter_right_y):

        self.place = place
        self.direction = direction
        self.counted_cars = []

        self.counter_left_x = counter_left_x
        self.counter_left_y = counter_left_y
        self.counter_right_x = counter_right_x
        self.counter_right_y = counter_right_y

    def count(self, centroids):
        for centroid in centroids:
            print(centroid)
            if centroid[0] > self.counter_left_x and centroid[0] < self.counter_right_x and centroid[
                1] > self.counter_left_y and centroid[1] < self.counter_right_y:
                self.counted_cars.append(datetime.datetime.now())

    if __name__ == '__main__':
        car_detector = CarDetector('http://88.212.15.20:5000/live/kamera_test_`janatarova/playlist.m3u8')
        counter_jantarova = Counter('palackeho', 'jantarova', 230, 450, 440, 550)

        while True:
            cars = car_detector.get_car_centroids()
            print(cars)
            counter_jantarova.count(cars)
            print(counter_jantarova.counted_cars)


#DATABASE CONNECTIONS WERE REMOVED FOR SAFETY REASONS




