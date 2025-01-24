# LSTM classification for human keypoints extract from Openpose
# Author: dean456
# Please add reference when using this code. Thanks

import os
import math
import numpy as np
import matplotlib.pyplot as plt
import json
from sklearn import preprocessing, impute
import tensorflow as tf
from keras_tuner import Hyperband

class Kinematics:
    def __init__(self):
        self.motion = {
            "frame": [],
            "time": [],
            "distance": [],
            "angle": [],
            "angularVel": [],
            "angularAcc": [],
            "frameLostData": [],
            "k1s": [],
            "k2s": [],
            "k3s": [],
            "k4s": []
        }

    @staticmethod
    def get_vector(k1, k2):
        if not k1 or not k2:
            return np.nan

        x = k2[0] - k1[0] if k1[0] and k2[0] else np.nan
        y = k2[1] - k1[1] if k1[1] and k2[1] else np.nan

        return [x, y]

    @staticmethod
    def get_unit_vector(k1, k2):
        vector = Kinematics.get_vector(k1, k2)
        norm = np.linalg.norm(vector)

        return vector / norm if norm > 0 else np.nan

    @staticmethod
    def get_distance(k1, k2):
        return np.linalg.norm(Kinematics.get_vector(k1, k2))

    @staticmethod
    def get_angle_coronal(k1, k2, k3, k4):
        try:
            u1 = Kinematics.get_unit_vector(k1, k2)
            u2 = Kinematics.get_unit_vector(k3, k4)
            return math.degrees(np.arccos(np.dot(u1, u2)))
        except:
            return np.nan

    def process_motion_coronal(self, k1s, k2s, k3s, k4s, frames, frame_rate=30):
        for f, (k1, k2, k3, k4, frame) in enumerate(zip(k1s, k2s, k3s, k4s, frames)):
            if all([k1, k2, k3, k4, frame]):
                angle = self.get_angle_coronal(k1, k2, k3, k4)
                distance = self.get_distance(k1, k2)

                self.motion["angle"].append(angle)
                self.motion["distance"].append(distance)
                self.motion["frame"].append(frame)
                self.motion["time"].append(f / frame_rate)
            else:
                self.motion["frameLostData"].append(frame)

class Stats:
    def __init__(self, data):
        self.data = {
            "data": data,
            "max": np.nanmax(data),
            "min": np.nanmin(data),
            "avg": np.nanmean(data),
            "median": np.nanmedian(data),
            "std": np.nanstd(data),
            "var": np.nanvar(data)
        }

    def moving_statistics(self, window_size, frame_rate):
        stats = {key: [] for key in ["max", "min", "avg", "median", "std", "var"]}

        for start in range(len(self.data["data"]) - window_size + 1):
            window = self.data["data"][start:start + window_size]
            for stat in stats:
                stats[stat].append(getattr(np, f"nan{stat}")(window))

        self.moving_data = {**stats, "time": np.arange(len(stats["max"])) / frame_rate}

class OpenposeIO:
    def __init__(self, datapath, jsonfoldername="jsons"):
        self.datapath = datapath
        self.jsonfoldername = jsonfoldername
        self.folders = [folder for folder in os.listdir(datapath) if os.path.isdir(os.path.join(datapath, folder)) and jsonfoldername in os.listdir(os.path.join(datapath, folder))]

    @staticmethod
    def listdir_no_hidden(path):
        return [f for f in os.listdir(path) if not f.startswith('.')]

    def extract_keypoints(self, subject):
        folder_path = os.path.join(self.datapath, subject, self.jsonfoldername)
        keypoint_files = self.listdir_no_hidden(folder_path)
        keypoints = {
            "Frames": [],
            "Nose": [],
            "Neck": [],
            "RShoulder": [],
            "RElbow": [],
            "RWrist": [],
            "LShoulder": [],
            "LElbow": [],
            "LWrist": [],
            "MidHip": [],
            "RHip": [],
            "RKnee": [],
            "RAnkle": [],
            "LHip": [],
            "LKnee": [],
            "LAnkle": [],
            "REye": [],
            "LEye": [],
            "REar": [],
            "LEar": [],
            "LBigToe": [],
            "LSmallToe": [],
            "LHeel": [],
            "RBigToe": [],
            "RSmallToe": [],
            "RHeel": []
        }

        for file in keypoint_files:
            with open(os.path.join(folder_path, file), 'r') as f:
                data = json.load(f)

                if data["people"]:
                    person_keypoints = data["people"][0]["pose_keypoints_2d"]
                    keypoints["Frames"].append(file)
                    keypoints["Nose"].append(person_keypoints[0:2])
                    keypoints["Neck"].append(person_keypoints[3:5])
                    keypoints["RShoulder"].append(person_keypoints[6:8])
                    keypoints["RElbow"].append(person_keypoints[9:11])
                    keypoints["RWrist"].append(person_keypoints[12:14])
                    keypoints["LShoulder"].append(person_keypoints[15:17])
                    keypoints["LElbow"].append(person_keypoints[18:20])
                    keypoints["LWrist"].append(person_keypoints[21:23])
                    keypoints["MidHip"].append(person_keypoints[24:26])
                    keypoints["RHip"].append(person_keypoints[27:29])
                    keypoints["RKnee"].append(person_keypoints[30:32])
                    keypoints["RAnkle"].append(person_keypoints[33:35])
                    keypoints["LHip"].append(person_keypoints[36:38])
                    keypoints["LKnee"].append(person_keypoints[39:41])
                    keypoints["LAnkle"].append(person_keypoints[42:44])
                    keypoints["REye"].append(person_keypoints[45:47])
                    keypoints["LEye"].append(person_keypoints[48:50])
                    keypoints["REar"].append(person_keypoints[51:53])
                    keypoints["LEar"].append(person_keypoints[54:56])
                    keypoints["LBigToe"].append(person_keypoints[57:59])
                    keypoints["LSmallToe"].append(person_keypoints[60:62])
                    keypoints["LHeel"].append(person_keypoints[63:65])
                    keypoints["RBigToe"].append(person_keypoints[66:68])
                    keypoints["RSmallToe"].append(person_keypoints[69:71])
                    keypoints["RHeel"].append(person_keypoints[72:74])

        return keypoints

# Define learning rate scheduler
def lr_scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

# Define hyperparameter tuning function
def build_model(hp):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
        units=hp.Int('units_layer1', min_value=32, max_value=128, step=32),
        return_sequences=True,
        dropout=hp.Float('dropout_layer1', min_value=0.1, max_value=0.5, step=0.1)
    )))
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
        units=hp.Int('units_layer2', min_value=32, max_value=128, step=32),
        dropout=hp.Float('dropout_layer2', min_value=0.1, max_value=0.5, step=0.1)
    )))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(4, activation='softmax'))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
        ),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

if __name__ == "__main__":
    # Example usage
    datapath = "path/to/data"
    subject = "example_subject"

    # Initialize OpenposeIO and extract keypoints
    op = OpenposeIO(datapath)
    keypoints = op.extract_keypoints(subject)

    # Example keypoints for motion processing
    neck = keypoints["Neck"]
    mid_hip = keypoints["MidHip"]
    left_knee = keypoints["LKnee"]
    left_ankle = keypoints["LAnkle"]
    frames = list(range(len(keypoints["Frames"])))

    # Initialize Kinematics and process motion
    kinematics = Kinematics()
    kinematics.process_motion_coronal(neck, mid_hip, left_knee, mid_hip, frames, frame_rate=30)

    # Display some results
    print("Motion Angles:", kinematics.motion["angle"][:5])
    print("Motion Distances:", kinematics.motion["distance"][:5])

    # Analyze stats
    stats = Stats(kinematics.motion["angle"])
    stats.moving_statistics(window_size=10, frame_rate=30)

    # Display moving statistics
    print("Moving Average of Angles:", stats.moving_data["avg"][:5])

    # Prepare data for LSTM
    X = np.array([kinematics.motion["angle"]])
    X = tf.keras.utils.pad_sequences(X, value=0, dtype='float32', padding='post')
    y = np.array([1])  # Placeholder target

    # Define model tuning
    tuner = Hyperband(
        build_model,
        objective='val_accuracy',
        max_epochs=20,
        factor=3,
        directory='my_dir',
        project_name='lstm_tuning'
    )

    # Early stopping
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

    # Search for best hyperparameters
    tuner.search(X, y, validation_split=0.2, epochs=20, callbacks=[callback])

    # Retrieve best model
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    best_model = tuner.hypermodel.build(best_hps)

    # Train best model
    history = best_model.fit(
        X, y,
        validation_split=0.2,
        epochs=50,
        batch_size=64,
        callbacks=[callback, tf.keras.callbacks.LearningRateScheduler(lr_scheduler)]
    )

    # Save the best model
    result_path = "path/to/results"
    best_model.save(result_path + '/optimized_lstm_model.h5')

    # Plot training history
    plt.figure()
    plt.title('Training and Validation Loss')
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig(result_path + '/optimized_train_loss.png')
