### 🚀 Intelligent Analysis and Training on the EuroSAT Dataset Using TensorFlow and Keras

<p align="center">
    <a href="https://github.com/M-RezaeiGH/TFDS_Tensorflow_Datasets_Keras_Eurosat_Dataset">
        <img src="https://github.com/user-attachments/assets/606f038d-0bc6-4937-8d74-faea00e0b886" alt="Project Badge">
    </a>
</p>

<p align="center">
    <a href="https://www.python.org">
        <img src="https://img.shields.io/badge/Python-3.10.7-blue.svg" alt="Python">
    </a>
    <a href="https://www.tensorflow.org">
        <img src="https://img.shields.io/badge/TensorFlow-2.12.0-orange.svg" alt="TensorFlow">
    </a>
    <a href="https://keras.io">
        <img src="https://img.shields.io/badge/Keras-2.12.0-red.svg" alt="Keras">
    </a>
    <a href="https://www.tensorflow.org/datasets">
        <img src="https://img.shields.io/badge/TFDS-EuroSAT-green.svg" alt="TFDS EuroSAT">
    </a>
</p>

---

## 👨‍💻 Developer

**Mohammad Reza Rezaei**  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue.svg)](https://www.linkedin.com/in/m-rezaei/)

---

## 🌟 Purpose and Objective

This project showcases a complete pipeline for loading, preprocessing, and training a deep learning model using the **EuroSAT dataset**. The goal is to classify satellite images into different land use and land cover categories using state-of-the-art deep learning techniques.

The project utilizes **TensorFlow**, **Keras**, and **TensorFlow Datasets (TFDS)** to build, train, and evaluate a convolutional neural network (CNN) for accurate predictions.

---

## 📊 Dataset Details

The **EuroSAT dataset** contains 27,000 labeled satellite images categorized into 10 classes, including forests, residential areas, agricultural lands, and more. Each image has dimensions of 64x64 pixels in RGB format.

---

## 🔧 Features of the Implementation

### Tasks Covered:
1. **Dataset Preparation**: Load the EuroSAT dataset from TensorFlow Datasets and split it into training, validation, and test sets.
2. **Data Preprocessing**: Resize images, rescale pixel values, and apply one-hot encoding to labels.
3. **Data Augmentation (Optional)**: Enhance model performance by applying transformations like random flipping, cropping, and brightness adjustments.
4. **Custom Training Loop**:
    - Shuffle, batch, cache, and prefetch data for optimal performance.
    - Define training and validation steps for custom loop execution.
5. **Model Architecture**:
    - Design a convolutional neural network (CNN) using the Keras functional API.
6. **Loss Function and Metrics**:
    - Use **categorical crossentropy** for loss and **categorical accuracy** as the evaluation metric.
7. **Training**:
    - Train the model with a custom loop that logs progress and validates performance.
8. **Evaluation**:
    - Assess model performance on validation and test sets.
    - Visualize predictions alongside ground truth labels.
9. **Visualization**:
    - Plot training and validation curves for loss and accuracy.

---

## 🚀 How to Run the Project

### 1. Clone the Repository:
```bash
git clone https://github.com/M-RezaeiGH/TFDS_Tensorflow_Datasets_Keras_Eurosat_Dataset.git
cd TFDS_Tensorflow_Datasets_Keras_Eurosat_Dataset
```

### 2. Install Required Libraries:
Ensure you have Python 3.10.7 installed, then run:
```bash
pip install -r requirements.txt
```

### 3. Run the Notebook:
Open the Jupyter Notebook `tensorflow_datasets_tfds_Keras_Eurosat_Dataset.ipynb`:
```bash
jupyter notebook tensorflow_datasets_tfds_Keras_Eurosat_Dataset.ipynb
```

---

## 📝 Key Libraries

- **TensorFlow**: Framework for deep learning.
- **Keras**: High-level API for building models.
- **Matplotlib**: Visualization library for plotting.
- **TensorFlow Datasets (TFDS)**: Provides prebuilt datasets like EuroSAT.

---

## 🛠️ Project Highlights

- Custom training loops provide flexibility and control over model training.
- Implements efficient data pipelines using TensorFlow's `map`, `batch`, `cache`, and `prefetch`.
- Augmentation techniques enhance model generalization capabilities.
- Real-time training progress visualization through loss and accuracy plots.

---

## 📈 Visualizations

- Training and validation loss and accuracy curves for performance tracking.
- Sample predictions compared to true labels for model evaluation.

---

## 📝 License

This project is licensed under the **Apache License 2.0**.

