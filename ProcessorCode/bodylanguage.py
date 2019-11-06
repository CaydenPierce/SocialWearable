import requests
import cv2
import time

#globals for for fps calculation
counter = 0
fpsTime = time.time()

#setup video stream (MJPG)
def setupCam(ip="192.168.1.10", port="8080"):
	cap = cv2.VideoCapture('http://{}:{}/?action=stream'.format(ip, port))
	return cap

#kill video stream
def killCam(cap):
	cap.release()

#receives video stream, computes fps, displays image
def getFrame(cap):
	global counter, fpsTime
	ret, frame = cap.read()
	if ret:
		cv2.imshow('Frame', frame)
		c = cv2.waitKey(10)
	else:
		return False
	counter += 1
	if counter >= 10:
		counter = 0
		fps = 10 / (time.time() - fpsTime)
		print("Current fps being read in is {} frames/seconds over last 10 frame. \r".format(fps), end="")
		fpsTime = time.time()

def sendAction(actionName, ip="192.168.1.10", port="8081"): #use this to send the wearable/pi a message describing the action we just saw
	resp = requests.post("http://{}:{}".format(ip, port), str(actionName))
	print(resp)

if __name__ == "__main__":
	cap = setupCam()
	sendAction("gang")
	while True:
		getFrame(cap)
	killCam(cap)

