from threading import Thread
import cv2
import time

class WebcamVideoStream:
	def __init__(self, src=0, name="WebcamVideoStream"):
		self.stream = cv2.VideoCapture(src)
		(self.grabbed, self.frame) = self.stream.read()
		self.src = src
		self.name = name

	def start(self):
		t = Thread(target=self.update, name=self.name, args=())
		t.daemon = True
		t.start()
		return self

	def update(self):
		while True:
			(self.grabbed, self.frame) = self.stream.read()
			if self.grabbed == False:
				print(f"Trying to reload camera {self.src}")
				self.stream = cv2.VideoCapture(self.src)
			# time.sleep(0.015)

	def read(self):
		return self.frame