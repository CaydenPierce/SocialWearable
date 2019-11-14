import requests
import cv2
import time
import poseinference
from poseinference import model_names
from time import sleep

#globals for for fps calculation
counter = 0
fpsTime = time.time()

#setup video stream (MJPG)
def setupCam(ip="192.168.1.2", port="8080"):
	cap = cv2.VideoCapture('http://{}:{}/?action=stream'.format(ip, port))
	return cap

#kill video stream
def killCam(cap):
	cap.release()

#receives video stream, computes fps, displays image
def getFrame(cap):
	global counter, fpsTime
	ret, frame = cap.read()
	"""if ret:
		cv2.imshow('Frame', frame)
		c = cv2.waitKey(10)
	else:
		return False
	counter += 1
	if counter >= 10:
		counter = 0
		fps = 10 / (time.time() - fpsTime)
		print("Current fps being read in is {} frames/seconds over last 10 frame. \r".format(fps))
		fpsTime = time.time()
	"""
	if ret == True:
		return frame
	else:
		return False

def sendAction(actionName, ip="192.168.1.2", port="8081"): #use this to send the wearable/pi a message describing the action we just saw
	resp = requests.post("http://{}:{}".format(ip, port), str(actionName))
	print(resp)

if __name__ == "__main__":
	cap = setupCam()
	import argparse
	parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
	# Model structure
	parser.add_argument('--arch', '-a', metavar='ARCH', default='hg',
	choices=model_names,
	help='model architecture: ' +
	' | '.join(model_names) +
	' (default: resnet18)')
	parser.add_argument('-s', '--stacks', default=8, type=int, metavar='N',
	help='Number of hourglasses to stack')
	parser.add_argument('-b', '--blocks', default=1, type=int, metavar='N',
	help='Number of residual modules at each location in the hourglass')
	parser.add_argument('--num-classes', default=16, type=int, metavar='N',
	help='Number of keypoints')
	parser.add_argument('--mobile', default=False, type=bool, metavar='N',
	help='use depthwise convolution in bottneck-block')
	parser.add_argument('--checkpoint', required=True, type=str, metavar='N',
	help='pre-trained model checkpoint')
	parser.add_argument('--in_res', required=True, type=int, metavar='N',
	help='input shape 128 or 256')
	parser.add_argument('--image', default='data/sample.jpg', type=str, metavar='N',
	help='input image')
	parser.add_argument('--device', default='cuda', type=str, metavar='N',
	help='device')

	model, in_res_h, in_res_w = poseinference.loadModel(parser.parse_args())

	timeCurr = time.time()
	while True:
		neck = False
		cvframe = getFrame(cap)
		image = poseinference.load_image(cvframe, in_res_h, in_res_w) #get frame, then load it in format to be sent through the model
		actionDetected, actionName = poseinference.main(parser.parse_args(), model, in_res_h, in_res_w, image, cvframe)
		#print("Time taken for one image: {}\r".format(time.time() - timeCurr))
		print(actionDetected)
		if actionDetected.any():
			sendAction(actionName)
		sleep(1)
		timeCurr = time.time()

	killCam(cap)
