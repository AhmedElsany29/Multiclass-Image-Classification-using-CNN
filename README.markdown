# Intel Image Classification with CNNs

## Project Overview

This project implements two convolutional neural network (CNN) models to classify images from the Intel Image Classification dataset into six categories: `buildings`, `forest`, `glacier`, `mountain`, `sea`, and `street`.

## Kaggle Notebook

You can view and run this project directly on Kaggle:  
[Multiclass Image Classification using CNN](https://www.kaggle.com/code/ahmedelsany/multiclass-image-classification-using-cnn)

The project compares two approaches:

1. **Model 1: Without Data Augmentation**  
   - Uses manually loaded images resized to 100x100 pixels.  
   - No data augmentation applied, making it faster but more prone to overfitting.  
   - Good for quick prototyping and baseline evaluation.

2. **Model 2: With Data Augmentation**  
   - Uses `ImageDataGenerator` for real-time augmentation (rotations, flips, zooms, etc.).  
   - Images resized to 150x150 pixels.  
   - Helps improve model generalization and robustness.

Both models use a VGG-like CNN architecture and are trained on the Intel Image Classification dataset from Kaggle.

---

## Dataset

- **Source:** [Intel Image Classification (Kaggle)](https://www.kaggle.com/datasets/puneet6060/intel-image-classification)  
- **Contents:**  
  - Training set: ~14,000 images in 6 categories  
  - Test set: ~3,000 images  
  - Prediction set: ~7,300 unlabeled images  
- **Preprocessing:**  
  - Model 1: Resize to 100x100, normalize pixel values to [0,1]  
  - Model 2: Resize to 150x150, apply augmentation and rescale

---

## How to Run

1. Download the dataset from Kaggle and unzip it locally.  
2. Adjust the dataset paths in the notebook if needed.  
3. Open the Jupyter Notebook and run cells sequentially to:  
   - Load and preprocess the data  
   - Train Model 1 (no augmentation) for 10 epochs  
   - Train Model 2 (with augmentation) for 30 epochs  
   - Evaluate and visualize results  

*Note:* Using a GPU is recommended for faster training.

---

## Libraries Used

- `tensorflow` (for building and training CNNs)  
- `numpy` (numerical operations)  
- `matplotlib` (visualization)  
- `opencv-python` (image loading for Model 1)  

Install libraries with:

**pip install tensorflow numpy matplotlib opencv-python**

---

## Comments on Models

| Model Type       | Description                                         | Purpose                          |
|------------------|-----------------------------------------------------|---------------------------------|
| **Basic Model**    | Trained on original images resized to 100x100 without augmentation. | Baseline for comparison.         |
| **Augmented Model**| Trained with real-time data augmentation on 150x150 images. | Improves generalization and reduces overfitting. |

---

## Potential Improvements

- For **Model 1**:  
  - Add light augmentation or increase dropout to reduce overfitting.  
  - Use `sparse_categorical_crossentropy` loss to simplify labels.  

- For **Model 2**:  
  - Fine-tune augmentation parameters (e.g., rotation range).  
  - Explore transfer learning with pre-trained models like VGG16 or ResNet.  

- General:  
  - Simplify or deepen the CNN architecture depending on performance.  
  - Apply cross-validation or hyperparameter tuning.

---


