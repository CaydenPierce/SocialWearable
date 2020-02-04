#creds to Daniil-Osokin  for the code in getPose() function (and of course for the sweet library, as mentioned in README)

#TODO get rid of hard coding, make cleaner, put it loop so if connection is dropped it waits for wearable to reconnect

import cv2
import sys

import requests
import time
from time import sleep
import torch
import numpy as np

#importing pose estimation library stuffs
sys.path.insert(1, '../lightweight-human-pose-estimation.pytorch') #this points to the pose estimation library
from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.keypoints import extract_keypoints, group_keypoints
from modules.load_state import load_state
from modules.pose import Pose, propagate_ids
from val import normalize, pad_width
from demo import infer_fast

#import body language movement extraction stuff
from BLmovements import BLmovements
from BLdecode import BLdecode

#import connection handler with wearable
from WearConn import WearConn

#import facial emotion stuff
sys.path.insert(1, './ResidualMaskingNetwork')
from ResidualMaskingNetwork import ssd_infer

#globals for fps calculation
counter = 0
fpsTime = time.time()

#setup video stream (MJPG)
def setupCam(ip="192.168.43.111", port="5000"):
    cap = cv2.VideoCapture('http://{}:{}/?action=stream'.format(ip, port))
    return cap

#kill video stream
def killCam(cap):
    cap.release()

#receives video stream, computes fps, displays image
def getFrame(cap):
    global counter, fpsTime
    ret, frame = cap.read()
    
    #update fps display
    counter += 1
    if counter >= 10:
        counter = 0
        fps = 10 / (time.time() - fpsTime)
        print("Current fps being read in is {} frames/seconds over last 10 frames. \r".format(fps))
        fpsTime = time.time()
    
    if ret == True:
        return frame
    else:
        return False

def sendAction(actionName, ip="192.168.1.2", port="8081"): #use this to send the wearable/pi a message describing the action we just saw
    resp = requests.post("http://{}:{}".format(ip, port), str(actionName))
    print(resp)

def getPose(net, img, stride, upsample_ratio):
    num_keypoints = Pose.num_kpts
    kpt_names = Pose.kpt_names
    previous_poses = []
    orig_img = img.copy()
    heatmaps, pafs, scale, pad = infer_fast(net, img, 256, stride, upsample_ratio, False)

    total_keypoints_num = 0
    all_keypoints_by_type = []
    for kpt_idx in range(num_keypoints):  # 19th for bg
            total_keypoints_num += extract_keypoints(heatmaps[:, :, kpt_idx], all_keypoints_by_type, total_keypoints_num)

    pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs, demo=True)
    for kpt_id in range(all_keypoints.shape[0]):
            all_keypoints[kpt_id, 0] = (all_keypoints[kpt_id, 0] * stride / upsample_ratio - pad[1]) / scale
            all_keypoints[kpt_id, 1] = (all_keypoints[kpt_id, 1] * stride / upsample_ratio - pad[0]) / scale
    current_poses = []
    pose_keypoints =None 
    for n in range(len(pose_entries)):
            if len(pose_entries[n]) == 0:
                    continue
            pose_keypoints = np.ones((num_keypoints, 2), dtype=np.int32) * -1
            for kpt_id in range(num_keypoints):
                    if pose_entries[n][kpt_id] != -1.0:  # keypoint was found
                            pose_keypoints[kpt_id, 0] = int(all_keypoints[int(pose_entries[n][kpt_id]), 0])
                            pose_keypoints[kpt_id, 1] = int(all_keypoints[int(pose_entries[n][kpt_id]), 1])
            #pose = Pose(pose_keypoints, pose_entries[n][18])
            #current_poses.append(pose)
            #pose.draw(img)
            #found_keypoints = pose.found_keypoints
    img = cv2.addWeighted(orig_img, 0.6, img, 0.4, 0)
    track_ids = True
    if track_ids == True:
            propagate_ids(previous_poses, current_poses)
            previous_poses = current_poses
            for pose in current_poses:
                    cv2.rectangle(img, (pose.bbox[0], pose.bbox[1]),
                                              (pose.bbox[0] + pose.bbox[2], pose.bbox[1] + pose.bbox[3]), (0, 255, 0))
                    cv2.putText(img, 'id: {}'.format(pose.id), (pose.bbox[0], pose.bbox[1] - 16),
                                            cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255))
    cv2.imshow('Lightweight Human Pose Estimation Python Demo', img)
    key = cv2.waitKey(33)
    if key == 27:  # esc
            return
    return pose_keypoints 

def connect():
    print("-Opening video stream...")
    stream = cv2.VideoCapture("/dev/stdin")
    print("-Stream opened")
    
    print("-Connecting to wearable")
    emex = WearConn(port=5000)
    print("-Wearable connected")

    return stream, emex

def loadEmotion():
    print("-Loading pose estimation neural net...")
    poseEstNet = PoseEstimationWithMobileNet()
    poseEstNet = poseEstNet.cuda()
    checkpoint = torch.load("../lightweight-human-pose-estimation.pytorch/checkpoint_iter_370000.pth", map_location='cpu')
    load_state(poseEstNet, checkpoint)
    print("-Pose estimation neural net loaded")
    
    print("-Opening body language decoders...")
    bodymove = BLmovements()
    bodydecode = BLdecode()
    print("-Body language decoder loaded")

    print("-Loading facial emotion neural net...")
    facialEmotionNet, image_size = ssd_infer.load()
    print("-Facial emotion neural net loaded")

    return poseEstNet, bodymove, bodydecode, facialEmotionNet, image_size


if __name__ == "__main__":
    poseEstNet, bodymove, bodydecode, facialEmotionNet, image_size = loadEmotion()
    stream, emex = connect()
    
    timeCurr = time.time()
    frames = 0
    lastUpdateTime = 0
    while True:
        try:
            frames += 1
            cvframe = getFrame(stream) #.read() 
            if cvframe is None:
                continue
            pose = getPose(poseEstNet, cvframe, 8, 4)
            if frames % 10 == 0:
                facialExp = ssd_infer.infer(cvframe, facialEmotionNet, image_size)
                print(facialExp)
            
            print("Streaming... Frame #{}".format(frames), end="\r")
            if pose is not None:
                bodymove.process(pose)
                print(bodymove.results)
                emotion = bodydecode.process(bodymove.results) 
                if (emotion > 0.6) and ((time.time() - lastUpdateTime) > 10):
                    stress = {"stress" : emotion}
                    emex.send(stress)
                    lastUpdateTime = time.time()
            timeCurr = time.time()
        except (ConnectionResetError, BrokenPipeError, Exception) as e:
            print(e)
            stream.release()
            emex.end()
            stream, emex = connect()
        except KeyboardInterrupt as e:
            print(e)
            stream.release()
            emex.end()
            print("Sock closed. Goodbye")
            break

    
