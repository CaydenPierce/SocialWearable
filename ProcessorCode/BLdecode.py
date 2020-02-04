#check out the book "What Every Body is Saying" by Joe Navarro. He shows how neck touches and face/head touches are stress depacification responses. Here we are count the frequency at which people are making these movement and prediction a stress level from it. This is powerful for strong personalities (*ahem*) who don't realize the intimidation / stress they create in others
import numpy as np

class BLdecode:
    def __init__(self):
        self.mNum = 70 #number of past actions considered
        self.necks = np.zeros(self.mNum, dtype=bool)
        self.heads = np.zeros(self.mNum, dtype=bool)
        self.state = {'stress' : 100}

    def process(self, data):
        self.checkNeck(data['neck'])
        self.checkHead(data['head'])

        moves = len(self.necks)

        stressMeter = (np.sum(self.necks) + np.sum(self.heads)) / moves

        return stressMeter

    def checkNeck(self, val):
        self.necks = np.roll(self.necks, 1)
        self.necks[0] = val
    
    def checkHead(self, val):
        self.heads= np.roll(self.heads, 1)
        self.heads[0] = val
