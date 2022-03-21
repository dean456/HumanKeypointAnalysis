import os
import json


class OpenposeIO:
    
    def __init__(self,datapath,jsonfoldername="jsons"):
        self.datapath=datapath # root path
        self.jsonfoldername=jsonfoldername
        self.folders=os.listdir(self.datapath)
        i=0
        while i>-1:
            if i<len(self.folders):
                if os.path.isdir(os.path.join(self.datapath,self.folders[i])):
                    subfolderList=os.listdir(os.path.join(self.datapath,self.folders[i]))
                    if jsonfoldername not in subfolderList:
                        del self.folders[i]
                    else:                    
                        i+=1
                else:
                    del self.folders[i]
            else:
                i=-1
        self.folderNum=len(self.folders)
               
    def extractKeypoints(self,subject):
        jsonfolderpath=os.path.join(self.datapath,subject,self.jsonfoldername)
        jsons=os.listdir(jsonfolderpath)
        k = Keypoints()
        for j in range(0,len(jsons)):
            if   j < 10:          index = "00000000000"+str(j)
            elif j < 100:         index = "0000000000"+str(j)
            elif j < 1000:        index = "000000000"+str(j)
            elif j < 10000:       index = "00000000"+str(j)
            elif j < 100000:      index = "0000000"+str(j)
            elif j < 1000000:     index = "000000"+str(j)
            elif j < 10000000:    index = "00000"+str(j)
            elif j < 100000000:   index = "0000"+str(j)
            elif j < 1000000000:  index = "000"+str(j)
            elif j < 10000000000: index = "00"+str(j)
            elif j < 100000000000:index = "0"+str(j)
            else:                 index = +str(j)          
            f = open(os.path.join(self.datapath,subject,self.jsonfoldername,subject+"_"+index+"_keypoints.json"))
            data = json.load(f) # DateType: Dictionary  
            for h in data['part_candidates']:
                k.Frames.append(index)
                k.Nose.append(h['0'])        
                k.Neck.append(h['1'])
                k.RShoulder.append(h['2'])
                k.RElbow.append(h['3'])
                k.RWrist.append(h['4'])
                k.LShoulder.append(h['5'])
                k.LElbow.append(h['6'])
                k.LWrist.append(h['7'])
                k.MidHip.append(h['8'])
                k.RHip.append(h['9'])
                k.RKnee.append(h['10'])
                k.RAnkle.append(h['11'])
                k.LHip.append(h['12'])
                k.LKnee.append(h['13'])
                k.LAnkle.append(h['14'])
                k.REye.append(h['15'])
                k.LEye.append(h['16'])
                k.REar.append(h['17'])
                k.LEar.append(h['18'])
                k.LBigToe.append(h['19'])
                k.LSmallToe.append(h['20'])
                k.LHeel.append(h['21'])
                k.RBigToe.append(h['22'])
                k.RSmallToe.append(h['23'])
                k.RHeel.append(h['24'])
        return k 
        
class Keypoints:
    def __init__(self):
        self.Frames,self.Nose,self.Neck,self.RShoulder,self.RElbow,self.RWrist,self.LShoulder,self.LElbow,self.LWrist,self.MidHip,self.RHip,self.RKnee,self.RAnkle,self.LHip,self.LKnee,self.LAnkle,self.REye,self.LEye,self.REar,self.LEar,self.LBigToe,self.LSmallToe,self.LHeel,self.RBigToe,self.RSmallToe,self.RHeel=[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]
        
        