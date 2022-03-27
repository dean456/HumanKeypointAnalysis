import os
import math
import numpy as np
import matplotlib.pyplot as plt
from openposeio import OpenposeIO, Keypoints
from kinematics import Kinematics, Stats, Plots
import platform
"""
"""
#Original videofolder: collect videonames
#videopath="/media/public/data/shoulderpain/video/"
datapath="/media/kang/Transcend/001_007" #1
op = OpenposeIO(datapath,"jsons")
imagefolder="images"
if not os.path.isdir(imagefolder):
    os.mkdir(imagefolder)
#default json folder is videoname/jsons if no input
framerate=30 # 30Hz
timewindow=30

X=[] # featureSet
y=[] # targetSet

for i in range(0,len(op.folders)):
    #Mark group #2
    subjectname=op.folders[i]
    tag = subjectname.split("_")
    group =tag[2][1]
    #pid = tag[0]
    #examdate = tag[1]
    #side = tag[2]
    #view = tag[3]
    #load = tag[4]
    print(tag)
   
    if (group=="1"):
        y.append(0) # control
    elif (group=="2"):
        y.append(1)
    
    #FeatureSet #3 
    k=op.extractKeypoints(subjectname) #keypoints
    hip=[]
    hip=Kinematics()
    hip.getMotionCorView(k.Neck,k.MidHip,k.LKnee,k.MidHip,k.Frames,framerate) # k1: distal k2: orgin k3:distal. k4:origin
    hipAstats = Stats(hip.motion["angle"])
    hipAstats.getMovStat(timewindow,framerate)    
    p=Plots.getMyPlot(1,imagefolder,"HipFlexion","Angle","Time","Degrees",subjectname,hip.motion["time"],hip.motion["angle"],'k-') #red "r-"
    
    hipDstats = Stats(hip.motion["distance"])
    hipDstats.getMovStat(timewindow,framerate)
    p=Plots.getMyPlot(1,imagefolder,"Trunk","Distance","Time","Distance",subjectname,hip.motion["time"],hip.motion["distance"],'k-')
    
    hipAAstats = Stats(hip.motion["angularAcc"])
    hipAAstats.getMovStat(90,framerate)
    p=Plots.getMyPlot(1,imagefolder,"HipFlex","AngularAcc","Time","N",subjectname,hip.motion["time"],hip.motion["angularAcc"],'k-')
    
    """
    if (side=="R"):
        if (view=="AP"):
            abd=Kinemiatics()
            abd.getMotionCorView(k.RElbow,k.RShoulder,k.MidHip,k.Neck,k.Frames,framerate)
            abdAstats = Stats(abd.motion["angle"])
            abdAstats.getMovStat(timewindow,framerate)
        elif(view=="LR"):
            abd=Kinematics()
            abd.getMotionNonCorView(k.RElbow,k.RShoulder,k.Frames,framerate)
            abdAstats = Stats(abd.motion["angle"])
            abdAstats.getMovStat(timewindow,framerate)
        
    elif (side=="L"):
        if (view=="AP"):
            abd=Kinematics()
            abd.getMotionCorView(k.LElbow,k.LShoulder,k.MidHip,k.Neck,k.Frames,framerate)
            abdAstats = Stats(abd.motion["angle"])
            abdAstats.getMovStat(timewindow,framerate)
        elif(view=="LR"):
            abd=Kinematics()
            abd.getMotionNonCorView(k.LElbow,k.LShoulder,k.Frames,framerate)
            abdAstats = Stats(abd.motion["angle"])
            abdAstats.getMovStat(timewindow,framerate)
    """
    #Kinematics features
    #p=Plots.getMyPlot(i,"Abduction","Angle","Frames","Degree",subjectname,abd.motion["time"][300:len(abd.motion["time"])-300],abd.motion["angle"][300:len(abd.motion["time"])-300],'k-')
    featureName = ["hipAmax","hipAmin","hipAavg","hipAstd","hipAvar","hipDmax","hipDmin","hipDavg","hipDstd","hipDvar","hipAAmovavg"]
    featureSet = [hipAstats.data["max"],
                  #hipAstats.data["min"],
                  #hipAstats.data["avg"],
                  #hipAstats.data["std"],
                  hipAstats.data["var"],
                  hipDstats.data["max"],
                  #hipDstats.data["min"],
                  #hipDstats.data["avg"],
                  #hipDstats.data["std"],
                  hipDstats.data["var"],
                  hipAAstats.movdata["avg"][1],
                  max(hipAAstats.movdata["max"])
                 ]
              #flexAstats.data["max"],flexAstats.data["min"],flexAstats.data["avg"],flexAstats.data["std"],flexAstats.data["var"]]
    isdelete=0
    # zero filling
    for i in range(0,len(featureSet)-1):
        if (math.isinf(featureSet[i])):
            print(i)
            print(featureSet[i])
            featureSet[i] = 0
            isdelete=1
        if (math.isnan(featureSet[i])):
            print(i)
            print(featureSet[i])
            featureSet[i] = 0
            isdelete=1
        if (featureSet[i]>np.finfo(np.float64).max):
            print(i)
            print(featureSet[i])
            featureSet[i] = 0
            isdelete=1
        if (featureSet[i]<np.finfo(np.float64).min):
            print(i)
            print(featureSet[i])
            featureSet[i] = 0
            isdelete=1

    if(isdelete==1):
        del y[-1]
    else:
        X.append(featureSet)

X= np.reshape(X,[len(op.folders),len(featureSet)])

#print(X)
#print(y)

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn import datasets
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

#iris = datasets.load_iris()
#X1 = iris.data[:, 0:2]  # we only take the first two features for visualization
#y1 = iris.target
#print(X1)
#print(y1)

n_features = X.shape[1]

C = 10
kernel = 1.0 * RBF(np.ones(len(featureSet)))  # for GPC

# Create different classifiers.
classifiers = {
    "L1 logistic": LogisticRegression(
        C=C, penalty="l1", solver="saga", multi_class="multinomial", max_iter=10000
    ),
    "L2 logistic (Multinomial)": LogisticRegression(
        C=C, penalty="l2", solver="saga", multi_class="multinomial", max_iter=10000
    ),
    "L2 logistic (OvR)": LogisticRegression(
        C=C, penalty="l2", solver="saga", multi_class="ovr", max_iter=10000
    ),
    "Linear SVC": SVC(kernel="linear", C=C, probability=True, random_state=0),
    "Gaussian Process": GaussianProcessClassifier(kernel),
    "Nearest Neighbors":KNeighborsClassifier(2), 
    "Linear SVM": SVC(kernel="linear", C=0.025),
    "RBF SVM":SVC(gamma=2, C=1),
    "Decision Tree":DecisionTreeClassifier(max_depth=5),
    "Random Forest":RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    "Neural Net":MLPClassifier(alpha=1, max_iter=1000),
    "AdaBoost": AdaBoostClassifier(),
    "Naive Bayes": GaussianNB(),
    "QDA": QuadraticDiscriminantAnalysis()   
}

n_classifiers = len(classifiers)

with open('report.txt','a+') as f:
    f.write("No Cross Validation\n")
    f.write("Feature Set:\n")
    for n in range(0,len(featureName)):
        f.write(featureName[n])
        f.write(",")
    f.write('\n\n')
    
    for index, (name, classifier) in enumerate(classifiers.items()):      
        classifier.fit(X, y)
        y_pred = classifier.predict(X)
        accuracy = accuracy_score(y, y_pred)
        report = classification_report(y,y_pred)
        print("Accuracy (train) for %s: %0.1f%% " % (name, accuracy * 100))
        f.write("Accuracy (train) for %s: %0.1f%% " % (name, accuracy * 100))
        f.write('\n')
        f.write(classification_report(y,y_pred))
        f.write('\n')
    f.write('The end of this trial\n')

    
from sklearn.model_selection import cross_val_score

with open('report.txt','a+') as f:
    f.write("Cross Validation\n")
    f.write("Feature Set:\n")
    for n in range(0,len(featureName)):
        f.write(featureName[n])
        f.write(",")
    f.write('\n\n')
    
    for idx, (name, clf) in enumerate(classifiers.items()):
        scores=cross_val_score(clf,X,y, cv=10)
        print("Accuracy for %s: %0.2f with a standard deviation of %0.2f (cross-validation)" % (name, scores.mean(), scores.std()))
        f.write("Accuracy for %s: %0.2f with a standard deviation of %0.2f (cross-validation)" % (name, scores.mean(), scores.std()))
        f.write('\n')

    f.write('The end of this trial')
    f.write('\n')
    