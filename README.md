# Concrete Crack Classification using VGG16 and ResNet50

# Introduction
This project focuses on building an image classifier to detect concrete cracks using deep learning. The models used are based on pre-trained architectures: **VGG16** and **ResNet50**. The objective is to compare their performances in classifying images of cracked and non-cracked concrete.

# Table of Contents
- [Dataset](#dataset)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Prediction](#prediction)
- [Results](#results)
- [References](#references)

# Dataset
The dataset used in this project is available at:
[Concrete Crack Dataset](https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DL0321EN/data/concrete_data_week4.zip)

After downloading and unzipping the dataset, the data is organized into three folders:
- **train/** (30,001 images)
- **valid/** (9,501 images)
- **test/** (500 images)

# Model Training
Two deep learning models were trained:
1. **VGG16-based classifier**
2. **ResNet50-based classifier**
You can find the models saved here --> https://drive.google.com/drive/folders/1yKgcIpBDqlrOVAlZol_8s6ICJVqTpuT6?usp=sharing

# Steps Involved:
- Used **ImageDataGenerator** for data preprocessing and augmentation.
- Loaded **VGG16** and **ResNet50** models with pre-trained ImageNet weights.
- Replaced the top layers with a **Dense** classification layer (2 classes: cracked, non-cracked).
- Used **Adam optimizer** and **categorical cross-entropy loss function**.
- Trained the models using `model.fit_generator()` with batch size **100** for **2 epochs**.

# Code Snippet for VGG16 Model:
```python
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input

num_classes = 2
batch_size = 100
image_size = (224, 224)

generator = ImageDataGenerator(preprocessing_function=preprocess_input)

training_generator = generator.flow_from_directory(
    "concrete_data_week4/train", target_size=image_size, batch_size=batch_size, class_mode="categorical"
)

model_vgg16 = Sequential()
model_vgg16.add(VGG16(include_top=False, pooling="avg", weights="imagenet"))
model_vgg16.add(Dense(num_classes, activation="softmax"))
model_vgg16.layers[0].trainable = False

model_vgg16.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model_vgg16.fit_generator(training_generator, epochs=2)
```

# Evaluation
Both models were evaluated on the test dataset:
```python
from keras.models import load_model

testing_generator = generator.flow_from_directory("concrete_data_week4/test", target_size=image_size, shuffle=False)

model_resnet50 = load_model("classifier_resnet_model.h5")
performance_vgg16 = model_vgg16.evaluate_generator(testing_generator)
performance_resnet50 = model_resnet50.evaluate_generator(testing_generator)
```
# Results:
- **VGG16 Model**
  - Loss: **0.00741**
  - Accuracy: **99.6%**
- **ResNet50 Model**
  - Loss: **0.11661**
  - Accuracy: **95.2%**

# Prediction
Predictions were made on the test set, and the first five predictions were displayed for both models.
```python
predictions_vgg16 = model_vgg16.predict_generator(testing_generator, steps=1)

def print_prediction(prediction):
    if prediction[0] > prediction[1]:
        print("Negative ({}% certainty)".format(round(prediction[0] * 100, 1)))
    elif prediction[1] > prediction[0]:
        print("Positive ({}% certainty)".format(round(prediction[1] * 100, 1)))
    else:
        print("Unsure (prediction split 50–50)")

print("First five predictions for the VGG16-trained model:")
for i in range(5):
    print_prediction(predictions_vgg16[i])
```

# Sample Predictions:
**VGG16 Model:**
- Negative (99.7% certainty)
- Negative (92.1% certainty)
- Negative (97.0% certainty)
- Negative (98.7% certainty)
- Negative (98.1% certainty)

**ResNet50 Model:**
- Negative (100.0% certainty)
- Negative (100.0% certainty)
- Negative (100.0% certainty)
- Negative (99.9% certainty)
- Negative (99.4% certainty)

# Results
- The **VGG16 model** outperformed **ResNet50**, achieving a higher accuracy.
- The classifier is effective in distinguishing cracked vs. non-cracked concrete.
- Fine-tuning deeper layers of ResNet50 could improve its performance.

# References
- [Keras Documentation](https://keras.io/)
- [VGG16 Paper](https://arxiv.org/abs/1409.1556)
- [ResNet Paper](https://arxiv.org/abs/1512.03385)

