
# CMPE 401 Project 1 — YOLO11 Object Detection on VisDrone
## Project Overview

This project investigates object detection using **YOLO11n** on the **VisDrone** dataset. The goal was to build a complete object detection workflow, analyze model training behaviour, and evaluate whether increasing image resolution improves detection performance.

The project consists of:

- Baseline YOLO11n training
- Training and validation loss analysis
- Overfitting and underfitting analysis
- A controlled image-resolution experiment
- Quantitative model comparison
- Model improvement analysis
- Best-model selection
- Example object detections

---

## Dataset

The **VisDrone** dataset was used because it contains challenging real-world aerial scenes with many small and densely packed objects.

Dataset split used by Ultralytics:

- Training images: **6,471**
- Validation images: **548**
- Test images: **1,610**

The dataset contains 10 object classes:

1. Pedestrian
2. People
3. Bicycle
4. Car
5. Van
6. Truck
7. Tricycle
8. Awning-tricycle
9. Bus
10. Motor

VisDrone is particularly challenging because objects such as pedestrians, bicycles, and vehicles can occupy a very small portion of the image.

---

## Model

The model used in this project was **YOLO11n**, the nano version of YOLO11.

YOLO11n was selected because it provides a lightweight and computationally efficient object detection model while still allowing the effects of different experimental settings to be studied.

---

## Experimental Setup

Two YOLO11n configurations were evaluated.

| Setting | Baseline | Experiment |
|---|---:|---:|
| Model | YOLO11n | YOLO11n |
| Dataset | VisDrone | VisDrone |
| Epochs | 25 | 25 |
| Batch Size | 8 | 8 |
| Image Resolution | 640 × 640 | 800 × 800 |
| Random Seed | 0 | 0 |
| max_det | 902 | 902 |

The main controlled variable was **image resolution**.

### Hypothesis

Increasing the input resolution from **640 × 640 to 800 × 800** was expected to improve detection performance because higher-resolution images preserve more spatial information about small objects.

All other major training settings were kept constant so that the effect of image resolution could be evaluated fairly.

---

## Part I — Baseline Model

The baseline model consisted of YOLO11n trained for **25 epochs at 640 × 640 resolution**.

Final baseline validation results:

| Metric | Result |
|---|---:|
| Precision | 0.4075 |
| Recall | 0.3106 |
| mAP50 | 0.2924 |
| mAP50-95 | 0.1619 |

---

## Part II — Loss Curve and Fitting Analysis

Training and validation losses were monitored over the 25 epochs.

The following were analyzed:

- Bounding-box loss
- Classification loss
- Distribution Focal Loss (DFL)
- mAP50
- mAP50-95

Both training and validation losses generally decreased during training, indicating that the model was learning.

There was no clear evidence of severe overfitting because validation loss did not begin increasing sharply while training loss continued decreasing.

There was also no strong evidence of severe underfitting because detection performance and mAP improved during training.

However, mAP was still improving near the end of the 25 epochs, suggesting that further training may potentially have produced additional improvement.

---

## Part III — Controlled Experiment

A controlled experiment was performed by increasing the image resolution from:

**640 × 640 → 800 × 800**

The model architecture, dataset, number of epochs, batch size, and random seed remained unchanged.

This modification was chosen because the VisDrone dataset contains many small objects, and increasing image resolution can preserve more detail before the images are processed by the neural network.

---

## Part IV — Iterative Model Improvement

The 800 × 800 model was evaluated against the original 640 × 640 baseline.

### Results

| Metric | 640 Baseline | 800 Experiment | Change |
|---|---:|---:|---:|
| Precision | 0.4075 | 0.4787 | +0.0712 |
| Recall | 0.3106 | 0.3575 | +0.0469 |
| mAP50 | 0.2924 | 0.3556 | +0.0632 |
| mAP50-95 | 0.1619 | 0.2029 | +0.0411 |

The 800-resolution model improved every major evaluation metric.

Approximate relative improvements were:

- Precision: **+17.5%**
- Recall: **+15.1%**
- mAP50: **+21.6%**
- mAP50-95: **+25.4%**

The largest relative improvement occurred in **mAP50-95**.

These results support the hypothesis that higher input resolution can improve object detection performance on a dataset containing many small objects.

The trade-off is that higher-resolution images require more computation during training and inference.

---

## Best Model

Based on validation **mAP50-95**, the best-performing configuration was:

**YOLO11n — 800 × 800 resolution**

Final performance:

- Precision: **0.4787**
- Recall: **0.3575**
- mAP50: **0.3556**
- mAP50-95: **0.2029**

This model was selected for the final detection examples.

---

## Example Detections

The selected 800-resolution model was tested on sample VisDrone validation images.

The model successfully detected multiple object types, including:

- Pedestrians
- People
- Cars
- Vans
- Trucks
- Bicycles
- Tricycles
- Awning-tricycles
- Motors

These examples demonstrate the operation of the final trained object detection pipeline on dense real-world scenes.

---

## Tools and Libraries

The project was completed using:

- Python
- Google Colab
- PyTorch
- Ultralytics
- YOLO11
- Pandas
- Matplotlib

Training and evaluation were performed using GPU acceleration.

---

## Running the Project

The complete implementation is contained in the Jupyter/Google Colab notebook included in this repository.

### 1. Install Ultralytics

```python
!pip install -U ultralytics
```

### 2. Import YOLO

```python
from ultralytics import YOLO
```

### 3. Load the baseline model

```python
model = YOLO("yolo11n.pt")
```

### 4. Train the baseline configuration

```python
model.train(
    data="VisDrone.yaml",
    epochs=25,
    imgsz=640,
    batch=8,
    seed=0,
    deterministic=True
)
```

### 5. Train the experimental configuration

```python
model = YOLO("yolo11n.pt")

model.train(
    data="VisDrone.yaml",
    epochs=25,
    imgsz=800,
    batch=8,
    seed=0,
    deterministic=True
)
```

Ultralytics automatically downloads and prepares the VisDrone dataset when `VisDrone.yaml` is used.

For validation in this project, `max_det=902` was used because some VisDrone images contain a very large number of objects.

---

## Conclusion

This project demonstrated a complete YOLO11 object detection workflow using the VisDrone dataset.

A YOLO11n baseline was first trained at 640 × 640 resolution. Training and validation curves were analyzed to investigate convergence, overfitting, and underfitting.

A controlled experiment then increased the input resolution to 800 × 800. The higher-resolution model improved precision, recall, mAP50, and mAP50-95.

The results indicate that increasing image resolution can significantly improve YOLO11n performance when detecting small and densely distributed objects. The 800 × 800 model was therefore selected as the final model.

Overall, the project demonstrates how controlled experimentation and quantitative evaluation can be used to improve a modern deep-learning object detection system.

---

## Repository Contents

- Jupyter/Google Colab notebook containing the complete project
- Training and validation analysis
- Controlled 640 vs 800 resolution experiment
- Quantitative comparison
- Detection examples
- Final project conclusions

---

**Course:** CMPE 401 — Deep Learning for Engineers  
**Project:** Instructor-Defined Project 1
