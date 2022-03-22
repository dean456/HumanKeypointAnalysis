import os
import numpy as np
import matplotlib.pyplot as plt
from openposeio import OpenposeIO, Keypoints
from kinematics import Kinematics, Stats, Plots
import platform
"""
"""
#Original videofolder: collect videonames
#videopath="/media/public/data/shoulderpain/video/"
datapath="/media/public/data/shoulderpain/video"
op = OpenposeIO(datapath,"jsons") #default json folder is videoname/jsons if no input
framerate=30 # 30Hz
timewindow=30

X=[]
y=[]

for i in range(0,len(op.folders)):
       
    subjectname=op.folders[i]
    print(subjectname)
    k=op.extractKeypoints(subjectname)
    if (subjectname[0]=="A"):
        y.append(1)
    elif (subjectname[0]=="R"):
        y.append(2)
    elif (subjectname[0]=="H"):
        y.append(0)
    
    if (subjectname[13]=="R"):
        if (subjectname[15]=="A"):
            abd=Kinematics()
            abd.getMotionCorView(k.RElbow,k.RShoulder,k.MidHip,k.Neck,k.Frames,framerate)
            abdAstats = Stats(abd.motion["angle"])
            abdAstats.getMovStat(timewindow,framerate)
        elif(subjectname[15]=="L"):
            abd=Kinematics()
            abd.getMotionNonCorView(k.RElbow,k.RShoulder,k.Frames,framerate)
            abdAstats = Stats(abd.motion["angle"])
            abdAstats.getMovStat(timewindow,framerate)
        
    elif (subjectname[13]=="L"):
        if (subjectname[15]=="A"):
            abd=Kinematics()
            abd.getMotionCorView(k.LElbow,k.LShoulder,k.MidHip,k.Neck,k.Frames,framerate)
            abdAstats = Stats(abd.motion["angle"])
            abdAstats.getMovStat(timewindow,framerate)
        elif(subjectname[15]=="L"):
            abd=Kinematics()
            abd.getMotionNonCorView(k.LElbow,k.LShoulder,k.Frames,framerate)
            abdAstats = Stats(abd.motion["angle"])
            abdAstats.getMovStat(timewindow,framerate)
    
    #Kinematics features
    p=Plots.getMyPlot(i,"Abduction","Angle","Frames","Degree",subjectname,abd.motion["time"][300:len(abd.motion["time"])-300],abd.motion["angle"][300:len(abd.motion["time"])-300],'k-')
    
    featureName = ["abdAmax","abdAmin","abdAavg","abdAstd","abdAvar"]
    featureSet = [abdAstats.data["max"],abdAstats.data["min"],abdAstats.data["avg"],abdAstats.data["std"],abdAstats.data["var"]]
              #flexAstats.data["max"],flexAstats.data["min"],flexAstats.data["avg"],flexAstats.data["std"],flexAstats.data["var"]]
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
        f.write(featureSetName[n])
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
        f.write(featureSetName[n])
        f.write(",")
    f.write('\n\n')
    
    for idx, (nm, clf) in enumerate(classifiers.items()):
        scores=cross_val_score(clf,X,y, cv=10)
        print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
        f.write("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
        f.write('\n')

    f.write('The end of this trial')
    f.write('\n')
    