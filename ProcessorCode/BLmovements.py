import math

class BL:
    def __init__(self, keypoints, neckThresh=40, headThresh=40):
        self.keypoints = keypoints
        self.neckThresh = headThresh
        self.headThresh = headThresh
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
        self.scaler = abs(torsoHeight / avgTorsoHeight)

    def neckTouch(self):
        leftHandDist = abs(self.distance(self.kpt_dict['l_wri'], self.kpt_dict['neck']))
        rightHandDist = abs(self.distance(self.kpt_dict['r_wri'], self.kpt_dict['neck']))
        #print(leftHandDist, type(leftHandDist))
        dist = min([leftHandDist, rightHandDist]) / self.scaler
        print("neck {}".format(dist))
        if dist < self.neckThresh:
            return True
        else:
            return False
    
    def headTouch(self):
        leftHandDist = abs(self.distance(self.kpt_dict['l_wri'], self.kpt_dict['nose']))
        rightHandDist = abs(self.distance(self.kpt_dict['r_wri'], self.kpt_dict['nose']))
        dist = min([leftHandDist, rightHandDist]) / self.scaler
        print("head {}".format(dist))
        if dist < self.headThresh:
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
        


