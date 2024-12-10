# Fashion MNIST Classification Using Convolutional Neural Networks (CNN)

## Overview

This project demonstrates a Deep Learning approach to classifying clothing items using a Convolutional Neural Network (CNN) trained on the Fashion MNIST dataset. By leveraging advanced neural network techniques, we've developed a model capable of accurately identifying various clothing categories with high precision.

The primary objective was to build a robust image classification model that can distinguish between 10 different types of clothing items. Our model achieved a training accuracy of 0.91 and a test accuracy of 0.88, showcasing its effectiveness in learning and generalizing clothing item features.

## Dataset

**Dataset**: Fashion MNIST (TensorFlow/Keras Datasets)

The Fashion MNIST dataset is a comprehensive collection of grayscale images representing 10 different clothing categories. Each image is a 28x28 pixel grayscale representation of a clothing item, making it an ideal benchmark for machine learning image classification tasks.

### Dataset Characteristics
- Total Images: 70,000
- Training Set: 60,000 images
- Test Set: 10,000 images
- Image Dimensions: 28x28 pixels
- Color: Grayscale
- Categories: 10 clothing types

## Model Architecture

Our Convolutional Neural Network (CNN) architecture is designed to effectively extract and learn features from clothing item images:

### Layer Configuration
1. **Convolutional Layer 1**: 30 filters, 3x3 kernel, ReLU activation
2. **Max Pooling Layer 1**: 2x2 pool size
3. **Convolutional Layer 2**: 60 filters, 3x3 kernel, ReLU activation
4. **Max Pooling Layer 2**: 2x2 pool size
5. **Flatten Layer**: Converts 2D feature maps to 1D feature vector
6. **Dense Layer 1**: 60 neurons, ReLU activation
7. **Output Layer**: 10 neurons (one per clothing category), Softmax activation

### Training Configuration
- **Optimizer**: Adam
- **Loss Function**: Sparse Categorical Cross-Entropy
- **Metrics**: Accuracy

## Results

### Performance Metrics
- **Training Accuracy**: 0.91
- **Test Accuracy**: 0.88

The model demonstrates strong performance in classifying clothing items, with a slight variance between training and test accuracy indicating good generalization.

## Dependencies

### Required Libraries
- Python 3.8+
- TensorFlow 2.x
- Keras
- NumPy
- Matplotlib


## How to Run

### Using Google Colab

1. Open Google Colab:
   - Navigate to [Google Colab](https://colab.research.google.com/)
   - Click on "New Notebook" or "File" > "New notebook"

2. Set Up Runtime:
   - Go to Runtime > Change runtime type
   - Select Python 3 as the runtime
   - (Optional) Select GPU for faster training

3. Install Dependencies:
   ```python
   !pip install tensorflow numpy matplotlib
   ```

4. Clone the Repository (if using GitHub):
   ```python
   !git clone https://github.com/Nirikshan95/Fashion-MNIST-Classification-Using-Deep-Learning.git
   ```

5. Open the Notebook:
   - Upload or open the Jupyter notebook for the project
   - Run cells sequentially
   - The dataset will be automatically downloaded via TensorFlow/Keras

### Local Development Alternative

If you prefer local development:

1. Clone the repository:
   ```bash
   git clone https://github.com/Nirikshan95/Fashion-MNIST-Classification-Using-Deep-Learning.git
   cd Fashion-MNIST-Classification-Using-Deep-Learning
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the Jupyter Notebook or Python scripts as needed

## Repository Structure
```
Fashion-MNIST-Classification-Using-Deep-Learning/
│
├── fashion_mnist_classification.ipynb             # notebook
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation
```

## Future Work

- Implement data augmentation techniques
- Experiment with more complex CNN architectures
- Explore transfer learning approaches
- Add real-world clothing item classification support

## License

### Dataset
The Fashion MNIST dataset used in this project is licensed under the [Creative Commons Attribution-ShareAlike 4.0 International License (CC BY-SA 4.0)](https://creativecommons.org/licenses/by-sa/4.0/).

## Contact

For questions or collaboration, please contact:
- Your Name
- Email: nirikshan987654321@gmail.com
- GitHub: @Nirikshan95
