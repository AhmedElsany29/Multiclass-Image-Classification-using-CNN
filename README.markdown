# Intel Image Classification with CNNs

## Project Overview
This repository contains a Jupyter Notebook that implements two convolutional neural network (CNN) models for classifying images from the Intel Image Classification dataset into six categories: `buildings`, `forest`, `glacier`, `mountain`, `sea`, and `street`. The project compares two approaches:

1. **Model 1: Without Data Augmentation**
   - A baseline CNN model with manually loaded images resized to `(100, 100)`.
   - No data augmentation, making it faster but less robust to overfitting.
   - Suitable for quick prototyping and baseline evaluation.

2. **Model 2: With Data Augmentation**
   - An advanced CNN model using `ImageDataGenerator` for real-time data augmentation.
   - Images resized to `(150, 150)` with transformations (rotation, flip, zoom, etc.).
   - Improves generalization, ideal for robust applications.

Both models use a VGG-like CNN architecture and are trained on the Intel Image Classification dataset from Kaggle.

## Dataset
- **Source**: [Intel Image Classification (Kaggle)](https://www.kaggle.com/datasets/puneet6060/intel-image-classification)
- **Structure**:
  - `seg_train/seg_train`: ~14,034 images across 6 categories (training).
  - `seg_test/seg_test`: ~3,000 images across 6 categories (testing).
  - `seg_pred/seg_pred`: 7,300 images for prediction (no labels).
- **Preprocessing**:
  - Model 1: Images resized to `(100, 100)`, normalized to `[0, 1]`.
  - Model 2: Images resized to `(150, 150)`, augmented, and rescaled.

## Repository Structure
```
intel-image-classification/
├── notebook.ipynb       # Jupyter Notebook with Model 1 and Model 2
├── README.md           # Project documentation
└── requirements.txt    # Python dependencies
```

## Prerequisites
- **Python**: 3.8 or higher
- **Dependencies**: Listed in `requirements.txt`
- **Hardware**: GPU recommended for faster training (e.g., NVIDIA GPU with CUDA support).
- **Dataset**: Download the Intel Image Classification dataset from Kaggle and place it in the `/kaggle/input/intel-image-classification/` directory (or adjust paths in the notebook).

## Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/<your-username>/intel-image-classification.git
   cd intel-image-classification
   ```

2. **Create a Virtual Environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the Dataset**:
   - Download the [Intel Image Classification dataset](https://www.kaggle.com/datasets/puneet6060/intel-image-classification) from Kaggle.
   - Place the unzipped dataset in the `/kaggle/input/intel-image-classification/` directory, or update the paths in the notebook to match your local directory structure.

## Usage
1. **Open the Notebook**:
   ```bash
   jupyter notebook notebook.ipynb
   ```

2. **Run the Notebook**:
   - Execute the cells sequentially to:
     - Load and preprocess the dataset.
     - Train Model 1 (without augmentation) and Model 2 (with augmentation).
     - Evaluate model performance on test/validation data.
     - Visualize training history and predictions (Model 2 only).
   - Ensure a GPU is available for faster training (configured to use `/GPU:0`).

3. **Key Sections**:
   - **Model 1**: Manual data loading, training for 10 epochs, no augmentation.
   - **Model 2**: Data augmentation with `ImageDataGenerator`, training for 30 epochs, prediction visualization.
   - Modify epochs, batch sizes, or augmentation parameters as needed.

## Requirements
See `requirements.txt` for a full list of dependencies. Key libraries include:
- `tensorflow` (for CNN models and training)
- `numpy`, `pandas` (data manipulation)
- `matplotlib` (visualization)
- `opencv-python` (image loading for Model 1)
- `kaggle` (optional, for dataset download via Kaggle API)

Install dependencies:
```bash
pip install tensorflow numpy pandas matplotlib opencv-python
```

## Results
- **Model 1**:
  - Test accuracy and loss reported after training.
  - Training history plotted (accuracy/loss curves).
- **Model 2**:
  - Validation accuracy and loss reported.
  - Visualizes predictions for 5 images from `seg_pred` dataset.
  - Expected to perform better due to data augmentation.

## Potential Improvements
- **Model 1**:
  - Use `sparse_categorical_crossentropy` to avoid one-hot encoding.
  - Add light augmentation or increase dropout to reduce overfitting.
- **Model 2**:
  - Fine-tune augmentation parameters (e.g., reduce rotation range).
  - Experiment with transfer learning (e.g., VGG16, ResNet) for better accuracy.
- **General**:
  - Simplify the CNN architecture to reduce parameters.
  - Add cross-validation or hyperparameter tuning.

## Contributing
Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make changes and commit (`git commit -m "Add feature"`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a pull request.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments
- Dataset provided by [Puneet6060 on Kaggle](https://www.kaggle.com/datasets/puneet6060/intel-image-classification).
- Built with TensorFlow and Keras for deep learning.

For issues or questions, open an issue on GitHub or contact the repository owner.