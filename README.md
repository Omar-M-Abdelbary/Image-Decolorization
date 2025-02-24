# Image Decolorization Using Deep Learning

## Overview
This project implements a deep learning model to decolorize images using a convolutional autoencoder. The model is trained using TensorFlow/Keras on a dataset of images and can be used to convert colored images to grayscale with enhanced details.

## Features
- Uses Convolutional Neural Networks (CNNs) for image decolorization.
- Implements an autoencoder architecture with encoder and decoder layers.
- Utilizes TensorFlow and Keras for model training and inference.
- Supports image augmentation using `ImageDataGenerator`.
- Uses OpenCV (`cv2`) for image preprocessing.

## Dependencies
Ensure you have the following libraries installed:
```bash
pip install numpy pandas opencv-python matplotlib scikit-learn tensorflow keras
```

## Dataset
The dataset is expected to be located at:
```
C:/Users/ghost/Desktop/tasks/dataset/dataset_updated/
```
Update the path in the script if needed.

## How to Run
1. Load the dataset and preprocess images.
2. Train the autoencoder model.
3. Use the trained model to decolorize test images.
4. Display results using `matplotlib`.

Run the Jupyter Notebook (`decolorize.ipynb`) step by step to execute the process.

## Results
The model takes colored images as input and outputs grayscale versions with enhanced features.

## Future Improvements
- Implement a more complex model for better decolorization quality.
- Train on a larger dataset for improved generalization.
- Add support for real-time image processing.

## Author
This project was developed as part of a deep learning exploration using autoencoders for image processing.

