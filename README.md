# Microplastic Detection in Water Bodies Using Machine Learning and Computer Vision Techniques ( YOLOV8 )

## Overview

This project focuses on detecting microplastics in environmental samples using a deep learning-based object detection model. Leveraging YOLOv8 and computer vision techniques, the system identifies microplastic particles in images with high accuracy and real-time performance.

---

## Problem Statement

Microplastics are a growing environmental concern, but detecting them manually is time-consuming and error-prone. This project aims to automate the detection process using AI to enable faster and scalable analysis.

---

## Approach

* Collected and preprocessed image datasets containing microplastics
* Applied data augmentation and preprocessing using OpenCV
* Trained a YOLOv8 object detection model using Ultralytics
* Evaluated model performance using mAP, precision, and recall
* Optimized inference pipeline for faster detection

---

## Tech Stack

* Languages: Python
* Libraries: YOLOv8 (Ultralytics), OpenCV, NumPy, Pandas
* Tools: Matplotlib, Jupyter Notebook

---

## Results

* Achieved ~85% mAP on test dataset
* Improved inference speed by ~25% after optimization
* Enabled real-time detection with bounding box predictions

---

## Project Structure

```
microplastics-detection/
│── data/                # Dataset (images & labels)
│── models/              # Trained YOLOv8 weights
│── notebooks/           # Training & evaluation notebooks
│── src/
│   ├── preprocessing.py
│   ├── train.py
│   ├── detect.py
│── results/             # Output predictions & metrics
│── README.md
```

---

## How to Run

### 1. Clone the repository

```
git clone https://github.com/your-username/microplastics-detection.git
cd microplastics-detection
```

### 2. Install dependencies

```
pip install -r requirements.txt
```

### 3. Train the model

```
python src/train.py
```

### 4. Run detection

```
python src/detect.py --source path/to/image
```

---

## Future Improvements

* Increase dataset size and diversity
* Deploy model using Flask or FastAPI for API-based inference
* Integrate with edge devices for real-time field applications
* Improve accuracy using advanced augmentation and tuning

---

## Contact

Dhanapraveen Krishna
Email: [rndpk12@gmail.com](mailto:rndpk12@gmail.com)
LinkedIn: (www.linkedin.com/in/rndpk)
