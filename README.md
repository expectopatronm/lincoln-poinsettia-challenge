![](documentation_files/Capture.PNG)

Lincoln Agri-Robotics (LAR) is the world’s first global centre of excellence in agricultural robotics, recently funded by UKRI’s Research England as part of their Expanding Excellence in England (E3) fund. This exciting centre bridges and expands the strong collaborations that exist between two leading research groups at the University of Lincoln: the Lincoln Institute for Agri-Food Technology (LIAT) and the Lincoln Centre for Autonomous Systems (L-CAS).

![](documentation_files/1.jpg)

AgriFoRwArdS is the world's first Centre for Doctoral Training (CDT) in Agri-Food Robotics. The Centre has been established by the University of Lincoln in collaboration with the University of Cambridge and the University of East Anglia.

![](documentation_files/AgriForwards.svg)

---
# Poinsietta Challenge

The poinsettia is a commercially important plant species of the diverse spurge family. Indigenous to Mexico and Central America, the poinsettia was first described by Europeans in 1834. It is particularly well known for its red and green foliage and is widely used in Christmas floral displays.

The Lincoln Agri-Robotics Poinsettia Challenge is a holiday-themed machine learning and computer
vision competition to devise intelligent ways to help identify features that contribute to rating the
"best" poinsettia. We will be looking at the height of the plant, colour of the leaves, and bushiness
among other attributes.

## 2nd Place Winner

---
## Find the 'bracts' challenge

![](documentation_files/poinsettia_test1.jpg)

The goal of the Find the bracts challenge is to find all the bracts (plant heads) in an image. This
could be used to estimate plant density, which is another criteria for grading poinsettia plants used
in nurseries.

## Solution

My solution consisted of two following stages:
1. Annotation corrections
2. Image augmentation
3. Transfer Learning

### Image augmentation

I decided to first augment the dataset of just 150 images to produce more diverse and varied frames which whould help in generalizing better. 
The various augmentation methods were as follows.

### Flip

Add horizontal or vertical flips to help your model be insensitive to subject orientation.

![](documentation_files/1.PNG)

### 90° Rotate
Add 90-degree rotations to help your model be insensitive to camera orientation.

![](documentation_files/2.PNG)

### Crop
Add variability to positioning and size to help your model be more resilient to subject translations and camera position.

![](documentation_files/3.PNG)

### Rotation
Add variability to rotations to help your model be more resilient to camera roll.

![](documentation_files/4.PNG)

### Shear
Add variability to perspective to help your model be more resilient to camera and subject pitch and yaw.

![](documentation_files/5.PNG)

### Grayscale
Probabilistically apply grayscale to a subset of the training set.

![](documentation_files/6.PNG)

### Hue
Randomly adjust the colors in the image.

![](documentation_files/7.PNG)

### Saturation
Randomly adjust the vibrancy of the colors in the images.

![](documentation_files/8.PNG)

### Brightness
Add variability to image brightness to help your model be more resilient to lighting and camera setting changes.

![](documentation_files/9.PNG)

### Exposure
Add variability to image brightness to help your model be more resilient to lighting and camera setting changes.

![](documentation_files/10.PNG)

### Blur
Add random Gaussian blur to help your model be more resilient to camera focus.

![](documentation_files/11.PNG)

## Neural Network Architecture
A preliminary comparision betweenthe state of the art FasterRCNN, EfficientNet and Yolov5
was conducted. A secondary distillation of the various Yolov5 architectures was conducted.

![](documentation_files/ffd.png)

![](documentation_files/yolov5.png)

![](documentation_files/yolo.png)

## Training Results
![](documentation_files/precision_recall_curve.png)

![](documentation_files/pr.png)

![](documentation_files/cf.PNG)

## Inference Results
![](documentation_files/test_batch1_pred.jpg)

---

![](documentation_files/unnamed.png)