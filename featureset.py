import os
import math
import numpy as np
from openposeio import Keypoints

class FeatureSet:

    def __init__(self):
        self.sets=0
        
class ShoulderFeature:
    def __init__(self):
        self.flexA=[]  #flexion angles
        self.flexAv=[] #flexion angular velocity
        self.flexAa=[] #flexion angular acceleration
        self.abdA=[]   #abduction
        self.abdAv=[]
        self.abdAa=[]
        self.hflexA=[] #horizontal flexion
        self.hflexAv=[]
        self.hflexAa=[]
        self.introtA=[] #internal rotation
        self.introtAv=[]
        self.introtAa=[]
        self.extrotA=[] #external rotation
        self.extrotAv=[]
        self.extrotAa=[]



class Kinematics:
    #def __init__(self): #k:Keypoint, v:vector
    
    def getVector(k1,k2):
        return [k2[0]-k1[0],k2[1]-k1[1]]
    
    def getUnitVector(k1,k2):
        v=Kinematics.getVector(k1,k2)
        return v/np.linalg.norm(v)
    
    def getDistance(k1,k2):
        return np.linalg.norm(Kinematics.getVector(k1,k2))
    
    def getAngle(k1,k2,k3,k4): # -90 < degree < 90
        return math.degrees(np.arccos(np.dot(Kinematics.getUnitVector(k1,k2),Kinematics.getUnitVector(k3,k4))))
        

    
    
        