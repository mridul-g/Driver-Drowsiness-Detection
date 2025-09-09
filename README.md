# Driver Drowsiness Detection

This project provides a lightweight, real-time driver drowsiness detection system using facial landmarks and a neural network model. The solution is optimized for small IoT devices by using TensorFlow Lite (TFLite), and has been successfully tested on a Raspberry Pi 3 with good accuracy.

## Overview

The system works by capturing video frames of the driver's face, extracting key facial features (Eye Aspect Ratio - EAR, Mouth Aspect Ratio - MAR), and classifying the driver's state (drowsy or alert) using a neural network. The model is trained and converted to TFLite for efficient inference on resource-constrained hardware.

## Workflow

1. **Data Collection**  
   - [`one.ipynb`](one.ipynb) collects video frames for both drowsy and normal states using a webcam.
   - Frames are saved in [`drowsy_frames/`](drowsy_frames/) and [`normal_frames/`](normal_frames/).

2. **Feature Extraction**  
   - EAR and MAR features are computed for each frame using dlib's facial landmark detector.
   - Features are saved to [`drowsy_features.csv`](drowsy_features.csv) and [`normal_features.csv`](normal_features.csv).

3. **Model Training**  
   - Features are loaded and normalized.
   - A small neural network is trained to classify drowsy vs. normal states.
   - The trained model is converted to TFLite ([`drowsiness_model.tflite`](drowsiness_model.tflite)).

4. **Real-Time Detection**  
   - [`two.ipynb`](two.ipynb) loads the TFLite model and performs real-time detection using webcam input.
   - If drowsiness is detected for a continuous period, an alert is displayed.

## Why TensorFlow Lite?

TensorFlow Lite enables fast, efficient inference on small IoT devices. By converting the model to TFLite, the system can run on devices like Raspberry Pi 3, making it suitable for in-vehicle deployment.

## File Descriptions

- [`one.ipynb`](one.ipynb):  
  - **Section 1:** Captures video frames for drowsy and normal states, saving them to disk.
  - **Section 2:** Extracts EAR and MAR features from each frame using dlib's 68-point facial landmark model ([`shape_predictor_68_face_landmarks.dat`](shape_predictor_68_face_landmarks.dat)).
  - **Section 3:** Saves extracted features to CSV files for both states.
  - **Section 4:** Trains a neural network on the features, evaluates accuracy, and converts the model to TFLite for deployment.

- [`two.ipynb`](two.ipynb):  
  - Loads the TFLite model ([`drowsiness_model.tflite`](drowsiness_model.tflite)).
  - Captures live video from the webcam.
  - Detects facial landmarks, computes EAR and MAR, and classifies the driver's state in real time.
  - If drowsiness is detected for a set duration, displays a visual alert.

- [`drowsy_features.csv`](drowsy_features.csv), [`normal_features.csv`](normal_features.csv):  
  - CSV files containing EAR, MAR, and label for each frame.

- [`drowsiness_model.tflite`](drowsiness_model.tflite):  
  - Optimized neural network model for drowsiness detection.

- [`shape_predictor_68_face_landmarks.dat`](shape_predictor_68_face_landmarks.dat):  
  - Dlib's pre-trained facial landmark model.

- [`drowsy_frames/`](drowsy_frames/), [`normal_frames/`](normal_frames/):  
  - Folders containing captured video frames for each state.

## Requirements

- Python 3.11+
- OpenCV
- Dlib
- TensorFlow / tflite-runtime
- Scipy, Numpy, Pandas

## Usage

1. Run [`one.ipynb`](one.ipynb) to collect data, extract features, train the model, and export to TFLite.
2. Run [`two.ipynb`](two.ipynb) for real-time drowsiness detection.

## Deployment

The TFLite model enables deployment on small IoT devices. The demo has been run on a Raspberry Pi 3, achieving good accuracy and real-time performance.

## Acknowledgements

Course Project for CS667.