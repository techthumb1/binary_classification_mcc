# Fruit Classification using Python, Machine Learning, and Deep Learning

This project is a fruit classification system that uses machine learning and deep learning to classify different types of fruits in real-time. The system can distinguish between apples, bananas, and oranges with high accuracy.

## Dataset

We used the Kaggle Fruit Recognition dataset to train the fruit classification model. The dataset contains 131 fruit types with a total of 90483 images. We selected three fruit types (apples, bananas, and oranges) to simplify the problem and focus on the classification accuracy.

## Preprocessing

We preprocessed the images to prepare them for machine learning. We resized the images to 64x64 pixels, converted them to grayscale, and normalized their pixel values to be between 0 and 1. We also converted the labels to one-hot encoding using the LabelEncoder and np_utils functions from the sklearn and keras libraries, respectively.

## Model Development

We used a deep convolutional neural network (CNN) to classify the fruit images. The model consisted of two convolutional layers, two max-pooling layers, one fully connected layer, one dropout layer, and one output layer. We used the Sequential model from the keras library to define the model architecture, and we compiled the model using the adam optimizer and the categorical_crossentropy loss function.

## Training

We split the dataset into training and testing sets with a ratio of 80:20. We trained the model on the training set using a batch size of 32 and 10 epochs. We also validated the model on the testing set to monitor its performance and prevent overfitting.

## Testing

We tested the fruit classification model on a new image of an apple. We loaded the image, preprocessed it, and used the trained model to predict its class. The model correctly predicted the class of the new image as an Apple Braeburn with high confidence.

## Conclusion

The fruit classification system using Python, machine learning, and deep learning can accurately classify different types of fruits in real-time. The system can be extended to classify other types of fruits or objects by adding more classes to the dataset and retraining the model. The system can also be improved by fine-tuning the model hyperparameters or using more advanced deep learning techniques.