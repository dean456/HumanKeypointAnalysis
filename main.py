import os
from openposeio import OpenposeIO, Keypoints
from featureset import Kinematics, ShoulderFeature, FeatureSet
"""
"""
#Original videofolder: collect videonames
#videopath="/media/public/data/shoulderpain/video/"
datapath="/home/kang/Documents/Aitools/shoulderpain/test"
op = OpenposeIO(datapath,"jsons") #default json folder is videoname/json
videoname="A10_20210901_L_AP_N"
k=op.extractKeypoints(videoname)

#kinematics
f=480
print("Left shoulder:"+str(k.LShoulder[f][0])+","+str(k.LShoulder[f][1]))
print("Left elbow:"+str(k.LElbow[f][0])+","+str(k.LElbow[f][1]))
print("Vector:"+str(Kinematics.getVector(k.LShoulder[f],k.LElbow[f])))
print("Distance:"+str(Kinematics.getDistance(k.LShoulder[f],k.LElbow[f])))
print("UnitVector:"+str(Kinematics.getUnitVector(k.LShoulder[f],k.LElbow[f])))
print("Angle:"+str(Kinematics.getAngle(k.LShoulder[f],k.LElbow[f],[0,0],[0,1])))


#Autosegment

#FeatureSet

#Analysed output
