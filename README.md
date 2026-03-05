# StrepClassification
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
---

## Project Overview and Description
Strep pharyngitis is an acute bacterial infection of the pharynx/tonsils caused by Streptococcus pyogenes (Group A Streptococcus, GAS). It is classified as an infectious pharyngitis, commonly presenting with abrupt fever, sore throat, and tonsillar exudates, often lacking cough or cold symptoms. The project focuses on classifying whether a patient has Strep pharyngitis using Deep Learning. The CNN model used is ResNet-18 followed by a Multi-Layer Perceptron.

## Model
The Convolutional Neural Network Model used is ResNet-18. The approach uses transfer learning to learning the throat features using the model. The last fully connected layer is removed so that the model outputs 512 features. For model training, every layer except the layer4 is frozen. The model trains the weights only of the layer4.   

![Alt text for the image](images/resnet18.png)

## Metrics
### Training
1. ROC-AUC
2. Train loss
3. Train accuracy
   
### Validation
1. ROC-AUC
2. Validation loss
3. Validation accuracy

## Dataset
There are two datasets in the repository, the Children National Hospital(cnh) and the Kaggle dataset.
1. Children National Hospital dataset:
This dataset contains 120 images and 7 symptoms associated with every image. The labels are either positive or negative. The label split is 1:1

2. (optional) Kaggle dataset
This dataset contains 300+ images seperated in different folders based on pharyngitis and no pharyngitis. The label split is 3:2

## Image Processing.
### White Balancing
White balancing is an image processing technique that removes unnatural color casts by adjusting the RGB channels so that neutral objects appear neutral white or gray. As the image is taken in the mouth, the redness of the mouth could be enhanced due to different lighting condition. To make the input image robust, we balance the image colors, so that we have consistent input to the model.

![Alt text for the image](images/white_balance.png)


### CLAHE
CLAHE (Contrast Limited Adaptive Histogram Equalization) is an image processing technique used to enhance the contrast in throat images, making it a valuable preprocessing step for automated strep throat classification. When combined with smartphone imaging and machine learning, this method helps to better highlight features like tonsillar exudate, inflammation, and red spots.

![Alt text for the image](images/CLAHE.png)

## Image Augmentations.
### Image Resize
The ResNet-18 generally expects image of dimension (224, 224). The input image from a dataset can be of different dimensions. Hence, it is a good practice to have standard image size.

### Image Random Crop
Crops a random portion of the image (70% to 100% area) and resizes it to IMG_RESIZE_DIM. This helps the model handle scale variance and partial occlusions.

### Image Random Rotation
Rotates the image by up to ±20 degrees to account for varying camera angles.
 
### Image Random Flip
Randomly flips images horizontally (50% chance) to double the spatial diversity.

### Image Color Jitter
Randomly adjusts brightness, contrast, and saturation (±30%) and hue (±5%) to make the model robust to different lighting conditions and sensor types.

### Image Random Blur
Applies a slight blur (30% probability) to simulate out-of-focus or motion-blurred inputs.

## Folder Directory
```text
+---data
|   +---cnh_dataset
|   +---csv
|   \---kaggle_dataset
|       +---test
|       \---train
|           +---no
|           \---phar
+---models
+---research
\---src
    +---common
    |   \---__pycache__
    +---pipeline
    |   \---__pycache__
    +---utils
    |   \---__pycache__
    \---__pycache__)
```

## Development Team
Ameya Konkar | Master of Engineering, Robotics | University of Maryland, College Park

## External Dependencies
- [Anaconda](https://anaconda.org/)
- [Opencv](https://github.com/opencv/opencv)
- [sklearn](https://scikit-learn.org/)
- [PyTorch](https://pytorch.org/)
- [Pandas](https://pandas.pydata.org/)

## Installation instructions
Install Anaconda on your Operating System

Create an Anaconda pseudo environment
```
conda create --name <env_name> python=3.10
conda activate <env_name>
```
Clone the git repo
```
cd <workspace>
sudo apt-get install git
git clone --recursive https://github.com/ameyakonk/StrepClassification.git
pip3 install -r requirements.txt
```

## Run instructions
(Note: Make sure the python interpretor is from the conda environment).
There are two datasets:
1. cnh_dataset  (Children's National Hospital)
2. (Optional) kaggle_dataset

#### Training Default: cnh
```
python3 train.py
```
(Optional)
for Kaggle dataset

```
python3 train.py --dataset kaggle
```

#### Evaluation
```
python3 eval.py
```





