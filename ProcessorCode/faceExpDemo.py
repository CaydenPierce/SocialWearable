import sys
import cv2
#import facial emotion stuff
sys.path.insert(1, './ResidualMaskingNetwork')
from ResidualMaskingNetwork import ssd_infer

cap = cv2.VideoCapture(0)
facialEmotionNet, image_size = ssd_infer.load()
while True:
    ret, cvframe = cap.read()
    if ret:
        cv2.imshow("frame", cvframe)
        cv2.waitKey(1)
        facialExp = ssd_infer.infer(cvframe, facialEmotionNet, image_size)
        print(facialExp)
