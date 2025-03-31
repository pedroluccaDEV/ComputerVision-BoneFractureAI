# Upper Limb Fracture Identifier

<p align="center">
  <img src="img/banner.jpg" alt="Banner" style="width: 100%; height: auto;">
</p>

## Idiomas:
<p align="center">
  <a href="README.pt.md" style="display: inline-block; padding: 10px 20px; font-size: 16px; text-align: center; background-color: #4CAF50; color: white; text-decoration: none; border-radius: 5px;">Português</a>
</p>

---
# Fracture Identifier in Upper Limbs

## Project Objective

This project aims to develop a computer vision model capable of identifying bone fractures in the upper limbs from radiographic images. Initially, the approach adopted was a binary classification model to determine whether a fracture was present or not. As the project progressed, fracture detection using the YOLO model was implemented, enabling not only classification but also the exact location of the fracture in the image.

## Project Evolution

### Phase 1: Binary Model

In the first phase, a convolutional neural network (CNN) model was trained to perform binary fracture detection. The model was fed with a dataset of radiographic images classified into two categories:

- **With fracture**
- **Without fracture**

From this model, it was possible to predict whether a new image contained a fracture or not. However, this approach had limitations as it did not provide information about the exact location of the fracture, making clinical interpretation difficult.

### Phase 2: YOLO Implementation for Fracture Detection

To overcome the limitations of the initial approach, the second phase of the project involved transitioning to a **YOLO (You Only Look Once)** model, which allows for precise detection and localization of fractures in images.

A new dataset, found on Kaggle, containing annotated images with fracture regions, was used. The dataset included three subdivisions (`train`, `test`, `val`) and annotations for seven categories of fractures:

- **Elbow Positive**
- **Fingers Positive**
- **Forearm Fracture**
- **Humerus Fracture**
- **Humerus**
- **Shoulder Fracture**
- **Wrist Positive**

The YOLO model was trained using this dataset, enabling it to not only detect the presence of fractures but also locate their position in the image.

## Maturation Process

### 1. Data Preparation

Before training, it was necessary to preprocess the images to ensure the data's quality. A preprocessing function was implemented using the **OpenCV** library:

```python
import cv2

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error loading image: {image_path}")
        return None

    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Histogram equalization to improve contrast
    equalized_image = cv2.equalizeHist(gray_image)
    
    # Apply smoothing filter to reduce noise
    blurred_image = cv2.GaussianBlur(equalized_image, (5, 5), 0)
    
    # Convert back to RGB for YOLO compatibility
    processed_image = cv2.cvtColor(blurred_image, cv2.COLOR_GRAY2BGR)
    
    return processed_image
```

### 2. YOLO Model Training

The YOLO model was trained using the structured dataset, ensuring it learned to identify different types of bone fractures in the upper limbs. The training required adjustments to hyperparameters, including the number of epochs and batch size, to optimize accuracy without compromising inference speed.

### 3. Detection and Classification Implementation

With the trained model, it was possible to develop a pipeline that performs the following steps:

1. Image loading and preprocessing.
2. Applying the YOLO model for fracture detection.
3. Displaying the regions affected by fractures with bounding boxes.
4. Displaying the name of the fracture class.
5. If no fractures are detected, the image is classified as "no fracture."

## Conclusion and Next Steps

The project evolved from simple binary classification to a robust system for detecting and localizing bone fractures in the upper limbs. Future improvements may include:

- Expanding the model to detect fractures in other parts of the body.
- Enhancing the dataset with expert-annotated images.
- Implementing a medical decision support system based on the model's detections.

This project represents a significant advancement in the use of computer vision for diagnostic support in radiology, contributing to faster and more accurate identification of bone fractures.
