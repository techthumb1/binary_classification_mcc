# Efficient Fruit Classification with Deep Learning

This project is a fruit classification system that uses machine learning and deep learning to classify different types of fruits in real-time. The system can distinguish between apples, bananas, and oranges with high accuracy.

## Requirements

- Python 3.6 or higher
- TensorFlow 2.x
- Keras 2.x
- NumPy
- Scikit-learn
- OpenCV

## Installation

1. Clone the repository to your local machine, <br>
`git clone https://github.com/your-username/fruit-classification-project.git`
2. To install the required packages, run the following command: <br>
`pip install -r requirements.txt`
3. Download the Kaggle Fruit Recognition dataset from [here](https://www.kaggle.com/moltean/fruits) and extract it to the project directory:
<https://www.kaggle.com/moltean/fruits>

## Usage

1. Preprocess the dataset by running the following command:<br>
`python preprocess.py`
2. Train the model by running the following command:<br>
`python train.py`
3. Test the model by running the following command:<br>
`python test.py`

## Dataset

We will use the Fruits-360 dataset, which consists of 90483 images of 131 different types of fruits. The images are of varying sizes and quality, and they were taken under different lighting and background conditions.

## Data Preparation

We will use Keras' ImageDataGenerator class to perform real-time data augmentation and preprocessing. This will help us to increase the size of our dataset, reduce overfitting, and improve the performance of our model.

## Model Architecture

We will use a deep convolutional neural network (CNN) to classify the fruit images. The model will consist of two convolutional layers, two max-pooling layers, one fully connected layer, one dropout layer, and one output layer.
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 62, 62, 32)        320       
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 31, 31, 32)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 29, 29, 64)        18496     
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 14, 14, 64)        0         
_________________________________________________________________
flatten (Flatten)            (None, 12544)             0         
_________________________________________________________________
dense (Dense)                (None, 128)               1605760   
_________________________________________________________________
dropout (Dropout)            (None, 128)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 3)                 387       
=================================================================
Total params: 1,624,963
Trainable params: 1,624,963
Non-trainable params: 0

