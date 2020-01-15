#creds to Daniil-Osokin as this is largely taken from the demo for the pose estimation library

import cv2
import sys
sys.path.insert(1, './streaming')
import StreamViewer as vidstream #our custom streaming stuff... way too slow but a bandaid for now
import requests
import time
from time import sleep
import torch
import numpy as np
import math

#importing pose estimation library stuffs
sys.path.insert(1, '../lightweight-human-pose-estimation.pytorch') #this points to the pose estimation library
from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.keypoints import extract_keypoints, group_keypoints
from modules.load_state import load_state
from modules.pose import Pose, propagate_ids
from val import normalize, pad_width
from demo import infer_fast

#globals for fps calculation
counter = 0
fpsTime = time.time()

#extracts body language meaning from pose estimation keypoints
"""for reference     kpt_names = ['nose', 'neck',
                 'r_sho', 'r_elb', 'r_wri', 'l_sho', 'l_elb', 'l_wri',
                 'r_hip', 'r_knee', 'r_ank', 'l_hip', 'l_knee', 'l_ank',
                 'r_eye', 'l_eye',
                 'r_ear', 'l_ear']
"""

class BL:
    def __init__(self, keypoints, touchThresh=100):
        self.keypoints = keypoints
        self.touchThresh = touchThresh
        self.kpt_dict = dict()
        kpt_names = ['nose', 'neck',
                 'r_sho', 'r_elb', 'r_wri', 'l_sho', 'l_elb', 'l_wri',
                 'r_hip', 'r_knee', 'r_ank', 'l_hip', 'l_knee', 'l_ank',
                 'r_eye', 'l_eye',
                 'r_ear', 'l_ear']
        for i, name in enumerate(kpt_names):
            self.kpt_dict[name] = keypoints[i] 
        self.results = self.pipeline()

    def getScale(self): #calculates a scaling factor based upon a very rough idea of how far away someone is
        avgHeight = 300 #170 #centimeter
        avgTorsoProportion = 3/8 #torso is 3 / 8ths of height, https://www.researchgate.net/figure/Proportions-of-the-Human-Body-with-Respect-to-the-Height-of-the-Head_fig4_228867476
        avgTorsoHeight = avgHeight * avgTorsoProportion #this doesn't actually mean anything in the real world until we map the camera output to real world numbers. Right now, just a number that will help us scale
        
        waistHeight = self.kpt_dict['r_hip'][1]
        neckHeight = self.kpt_dict['neck'][1]
        torsoHeight = neckHeight - waistHeight
        self.scaler = torsoHeight / avgTorsoHeight

    def neckTouch(self):
        leftHandDist = self.distance(self.kpt_dict['l_wri'], self.kpt_dict['neck'])
        rightHandDist = self.distance(self.kpt_dict['r_wri'], self.kpt_dict['neck'])
        #print(leftHandDist, type(leftHandDist))
        dist = min([leftHandDist, rightHandDist]) / self.scaler
        return dist
        if dist < self.touchThresh:
            return True
        else:
            return False
    
    def headTouch(self):
        leftHandDist = self.distance(self.kpt_dict['l_wri'], self.kpt_dict['nose'])
        rightHandDist = self.distance(self.kpt_dict['r_wri'], self.kpt_dict['nose'])
        dist = min([leftHandDist, rightHandDist]) / self.scaler
        return dist
        if dist < self.touchThresh:
            return True
        else:
            return False

    def distance(self, p1, p2):
        x = p1[0] - p2[0]
        y = p1[1] - p2[1]
        dist = math.sqrt(x ** 2 + y ** 2)
        return dist

    def pipeline(self): #pipe together body language movement decoders
        self.getScale() #figures out relative scale of person, distance from camera
        neck = self.neckTouch()
        head = self.headTouch()
        result = {'neck' : neck, 'head' : head, 'scale' : self.scaler}
        return result
        

#setup video stream (MJPG)
def setupCam(ip="192.168.43.111", port="3000"):
    cap = cv2.VideoCapture('http://{}:{}/?action=stream'.format(ip, port))
    return cap

#kill video stream
def killCam(cap):
    cap.release()

#receives video stream, computes fps, displays image
def getFrame(cap):
    global counter, fpsTime
    ret, frame = cap.read(cv2.IMREAD_COLOR)
    """if ret:
        cv2.imshow('Frame', frame)
        c = cv2.waitKey(10)
    else:
        return False"""
    
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
        """if track_ids == True:
                propagate_ids(previous_poses, current_poses)
                previous_poses = current_poses
                for pose in current_poses:
                        cv2.rectangle(img, (pose.bbox[0], pose.bbox[1]),
                                                  (pose.bbox[0] + pose.bbox[2], pose.bbox[1] + pose.bbox[3]), (0, 255, 0))
                        cv2.putText(img, 'id: {}'.format(pose.id), (pose.bbox[0], pose.bbox[1] - 16),
                                                cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255))"""
        """cv2.imshow('Lightweight Human Pose Estimation Python Demo', img)
        key = cv2.waitKey(33)
        if key == 27:  # esc
                return"""
        return pose_keypoints 

if __name__ == "__main__":
        print("-Loading neural net...")
        net = PoseEstimationWithMobileNet()
        net = net.cuda()
        checkpoint = torch.load("../lightweight-human-pose-estimation.pytorch/checkpoint_iter_370000.pth", map_location='cpu')
        load_state(net, checkpoint)
        print("-Neural net loaded")

        print("-Opening video stream...")
        stream = vidstream.StreamViewer('3000')
        print("-Stream opened")

        timeCurr = time.time()
        frames = 0
        while True:
            frames += 1
            stream.receive_stream()
            cvframe = stream.current_frame #getFrame(cap)
            pose = getPose(net, cvframe, 8, 4)
            if pose is not None:
                movements = BL(pose).results
                print(movements)
            #print(pose)
            #print("Recved frame")
            #print("FPS: {}".format(1/(time.time() - timeCurr)))
            timeCurr = time.time()

        #killCam(cap)
