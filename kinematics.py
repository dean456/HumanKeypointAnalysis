import os
import math
import numpy as np
from openposeio import Keypoints
import matplotlib.pyplot as plt
        
class Kinematics:
    def __init__(self):
        self.motion = {
            "frame": [], # real No. frame in the video
            "time": [], # time data for plot
            "distance": [],
            "angle": [],
            "angularVel": [],
            "angularAcc": [],
            "frameLostData": [],
        }
     
    def getVector(k1,k2):
        if not bool(k1):
            return [0,0]
        elif not bool(k2):
            return [0,0] 
        else: 
            if (k1[0] and k2[0]):
                x = k2[0]-k1[0]
            else:
                x = 0
            if (k1[1] and k2[1]):
                y = k2[1]-k1[1]
            else:
                y = 0
            return [x,y]
              
    def getUnitVector(k1,k2):
        v=Kinematics.getVector(k1,k2)
        if (np.linalg.norm(v) > 0):
            return v/np.linalg.norm(v)
        else:
            return 0
    
    def getDistance(k1,k2):
        #return math.sqrt(pow(k1[0]-k2[0],2)+pow(k1[1]-k2[1],2))
        return np.linalg.norm(Kinematics.getVector(k1,k2))
    
    def getAngleCorView(k1,k2,k3,k4): # Cor: coronal view; Angle: -90 < degree < 90
        return math.degrees(np.arccos(np.dot(Kinematics.getUnitVector(k1,k2),Kinematics.getUnitVector(k3,k4))))
    
    def getAngleNonCorView(k1,k2,maxLen): # NonCor: non coronal view = sagittal view or axial view. Angle: -90 < degree < 90
        d = Kinematics.getDistance(k1,k2)
        if (maxLen > 0 and d <= maxLen):
            return math.degrees(np.arccos(d/maxLen))
        else:
            return 0
    
    def getMotionCorView(self,k1s,k2s,k3s,k4s,frames,frameRate=30): #default framerate =30
        lostframe = 0
        for f in range(0,len(frames)):
            if (k1s[f] and k2s[f] and k3s[f] and k4s[f] and frames[f]): #is not empty
                angle = Kinematics.getAngleCorView(k1s[f],k2s[f],k3s[f],k4s[f])
                distance = Kinematics.getDistance(k1s[f],k2s[f])
                if (f > 1):
                    if (lostframe == f):
                        self.motion["angularVel"].append(0.0)
                        self.motion["angularAcc"].append(0.0)
                    else: 
                        angularV = (angle-self.motion["angle"][-1])*frameRate
                        self.motion["angularAcc"].append((angularV-self.motion["angularVel"][-1])*frameRate)
                        self.motion["angularVel"].append(angularV) 
                elif (f == 1):
                    if (lostframe == f):
                        self.motion["angularVel"].append(0.0)
                    else: 
                        angularV = (angle-self.motion["angle"][0])*frameRate
                        self.motion["angularVel"].append(angularV)
                    self.motion["angularAcc"].append(0.0)
                elif (f == 0):
                    self.motion["angularVel"].append(0.0)
                    self.motion["angularAcc"].append(0.0)
                self.motion["distance"].append(distance)
                self.motion["angle"].append(angle)
                self.motion["frame"].append(frames[f])
                self.motion["time"].append(f/frameRate) # unit:second
            else:
                self.motion["frameLostData"].append(frames[f])
                lostframe = len(self.motion["frameLostData"])
    
    def getMotionNonCorView(self,k1s,k2s,frames,frameRate=30): #default framerate =30
        maxLen  = 0.0
        distance = []
        lostframe=0
        for f in range(0,len(frames)): 
            distance.append(Kinematics.getDistance(k1s[f],k2s[f]))
        std = np.std(distance)
        mean = np.mean(distance)
        for f in range(0,len(frames)):
            if (maxLen < distance[f] and distance[f] < mean+2*std):
                maxLen = distance[f]
        for f in range(0,len(frames)):            
            if (k1s[f] and k2s[f] and frames[f]): #is not empty
                distance = Kinematics.getDistance(k1s[f],k2s[f])
                angle = Kinematics.getAngleNonCorView(k1s[f],k2s[f],maxLen)
                if (f > 1):
                    if (lostframe == f):
                        self.motion["angularVel"].append(0.0)
                        self.motion["angularAcc"].append(0.0)
                    else: 
                        angularV = (angle-self.motion["angle"][-1])*frameRate
                        self.motion["angularAcc"].append((angularV-self.motion["angularVel"][-1])*frameRate)
                        self.motion["angularVel"].append(angularV)                   
                elif (f == 1):
                    if (lostframe == f):
                        self.motion["angularVel"].append(0.0)
                    else: 
                        angularV = (angle-self.motion["angle"][0])*frameRate
                        self.motion["angularVel"].append(angularV)
                    self.motion["angularAcc"].append(0.0)
                elif (f == 0):
                    self.motion["angularVel"].append(0.0)
                    self.motion["angularAcc"].append(0.0)
                self.motion["distance"].append(distance)
                self.motion["angle"].append(angle)
                self.motion["frame"].append(frames[f])
                self.motion["time"].append(f/frameRate) # unit:second
            else:
                self.motion["frameLostData"].append(frames[f])
                lostframe = len( self.motion["frameLostData"])
    
    def getResampleData(data,newLen):
        interval=math.floor((len(data)-1)/newLen)
        resample=[]
        if (interval>0):
            for i in range(1,newLen):
                resample.append(data[interval*i])
        return resample
        

                
class Stats:
    def __init__(self,data): 
        self.data = {
            "data": data,
            "max": max(data), # real No. frame in the video
            "min": min(data), # time data for plot
            "avg": np.mean(data),
            "median": np.median(data),
            "std": np.std(data),
            "var": np.var(data)
        }        
        self.movdata = {
            "timewindow":[],
            "frame":[],
            "time":[],
            "max": [], # real No. frame in the video
            "min": [], # time data for plot
            "avg": [],
            "median": [],
            "std": [],
            "var": [],
        }
        
    
    def getMovStat(self,timewindow,framerate):
        self.movdata["timewindow"].append(timewindow)
        if (len(self.data["data"])>timewindow):
            for f in range(0,len(self.data["data"])-timewindow):
                self.movdata["frame"].append(f)
                self.movdata["time"].append(f/framerate)                                        
                self.movdata["max"].append(max(self.data["data"][f:f+timewindow-1]))
                self.movdata["min"].append(min(self.data["data"][f:f+timewindow-1]))
                self.movdata["avg"].append(np.mean(self.data["data"][f:f+timewindow-1]))
                self.movdata["median"].append(np.median(self.data["data"][f:f+timewindow-1]))
                self.movdata["std"].append(np.std(self.data["data"][f:f+timewindow-1]))
                self.movdata["var"].append(np.var(self.data["data"][f:f+timewindow-1]))
        else:
            return "Data length less than time window"

class Plots:
    def getMyPlot(fig,imagefolder,title,legend,labelx,labely,subjectname,x1,y1,c1='k-',x2=[],y2=[],c2=[]):
        plt.figure(fig)
        plt.plot(x1,y1,'k-')
        plt.legend(legend,loc="best")#,bbox_to_anchor=(0.2,0.2,0.5,0.5)) # loc="lower right, "bbox(x,y,width,height)
        plt.title(title+":"+subjectname)
        plt.xlabel(labelx)
        plt.ylabel(labely)
        plt.grid()
        plt.savefig(imagefolder+"/"+subjectname+"_"+title+".png") # default png
        #plt.show()
        plt.close()
        