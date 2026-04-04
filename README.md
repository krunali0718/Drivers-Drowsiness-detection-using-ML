# 🚗 Advanced Driver Drowsiness Detection System

![Drowsiness Project](https://img.shields.io/badge/Status-Active-brightgreen.svg)
![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-orange.svg)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-XGBoost%20%7C%20Random%20Forest-red.svg)

## 📌 Project Overview
The **Advanced Driver Drowsiness Detection System** is a real-time, Machine Learning-based computer vision application designed to prevent road accidents caused by driver fatigue. It actively monitors facial landmarks to measure Eye Aspect Ratio (EAR), Mouth Aspect Ratio (MAR), and Head Tilt, successfully warning the driver before they fall asleep.

This repository contains both a high-accuracy predictive ML model and an interactive **Streamlit Dashboard** for visualization and real-time inference.

## ✨ Features
* **Real-time Video Processing:** Monitors live camera feeds to detect fatigue.
* **Advanced Feature Extraction:** Uses Dlib/OpenCV to track 68 facial landmark points, computing blink rates, yawns, and head drops.
* **High-Accuracy ML Model:** Uses powerful tree-based models (XGBoost/Random Forest) with SMOTE for balanced class prediction.
* **Interactive Dashboard:** Built with Streamlit (`dashboard.py`) to visualize analytics, real-time metrics, and model performance.
* **Extensive EDA:** Demonstrates clear workflows inside Jupyter Notebooks for academic review and exploratory data analysis.

## 📂 Repository Structure

* **`master_app.py` / `dashboard.py`**: The main Streamlit web application.
* **`advanced_drowsiness_test.py`**: A python script strictly for running the real-time OpenCV webcam drowsiness tests.
* **`extract_features.py`**: Utilities to extract EAR, MAR, MOE, etc., from image datasets.
* **`train_model.py` / `tmp_tune.py`**: Scripts used for training and tuning predictive models using the extracted features.
* **`*.ipynb`**: Interactive notebooks (`Accident_EDA_and_Training.ipynb`) containing visual graphs, data balancing, and algorithmic proofs.
* **`run_project.bat`**: A quick-start batch script for Windows users to launch the application effortlessly.

*(Note: Extremely large datasets and generated Model `.pkl` files are ignored in this repository due to GitHub size constraints.)*

## 🚀 How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/krunali0718/Drivers-Drowsiness-detection-using-ML.git
cd Drivers-Drowsiness-detection-using-ML
```

### 2. Install Dependencies
You need Python installed. Install the requisite libraries using:
```bash
pip install -r requirements.txt
```
*(If you do not have a requirements.txt, you typically need to install `opencv-python`, `dlib`, `scikit-learn`, `xgboost`, `streamlit`, `pandas`, and `numpy`)*

### 3. Start the Web Dashboard
Launch the interactive Streamlit dashboard by running:
```bash
streamlit run dashboard.py
```
*(Or simply double-click the `run_project.bat` file if you are on Windows)*

### 4. Run Live Webcam Detection Directly
To just launch the real-time OpenCV camera detection window without the web dashboard:
```bash
python advanced_drowsiness_test.py
```

## 🧠 Methodology
1. **Facial Landmark Detection:** Localizes eyes and mouth using Dlib's pre-trained HOG + Linear SVM object detector.
2. **Feature Calculation:**
   * **EAR (Eye Aspect Ratio):** To detect micro-sleeps and prolonged blinks.
   * **MAR (Mouth Aspect Ratio):** To identify yawning.
   * **MOE (Mouth over Eye):** A combined metric.
   * **Head Pose Estimation:** To flag dangerous head drops or continuous tilting.
3. **Classification:** Features are fed into the tuned Machine Learning model. If metrics fall below safe thresholds progressively, a visual and alarming alert triggers.

## 🤝 Contribution
Created for academic research and presentation. Feel free to fork and improve!
