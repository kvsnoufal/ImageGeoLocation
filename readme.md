# Photo Geolocation with Neural Networks
Building a neural network that predicts that can geotag an outdoor image and how to catch a cheating neural network with grad-cam



Pls checkout the [medium article](https://medium.com/@noufalsamsudin/photo-geolocation-with-neural-networks-how-to-and-how-not-to-8aa7f10abb34) for a quick overview.


To train model:
```
python train.py
```


## Problem Statement

Building a CNN model geotag an image - take an image as input and predict the location of that image as output.

The model was trained on a dataset of google streetview images. I scraped images of random locations in India for generating this dataset. The model is reasonably good in making predictions. It generally predicts in the vicinity of the actual location.

![Pic of results](https://github.com/kvsnoufal/ImageGeoLocation/blob/main/docs/1.JPG)


These are handpicked good examples. Even when the model's predicted location is wrong, the predicted grids are reasonable:

![Pic of results2](https://github.com/kvsnoufal/ImageGeoLocation/blob/main/docs/2.JPG)


I cropped out the bottom portion of the image. The model accuracy was not as great, but it was able to pick up some general patterns like landscape, buildings, vegetation, roads, terrain etc when making the prediction.


![Pic of results2](https://github.com/kvsnoufal/ImageGeoLocation/blob/main/docs/3.JPG)

## Dataset Preparation

I overlayed an isometric grid onto the map of India. The resulting grids where the target variables the model needed to predict.

![Pic of results4](https://github.com/kvsnoufal/ImageGeoLocation/blob/main/docs/4.JPG)

I then uniformly sampled points in each grid, used google's streetview API to get the nearest location with a streetview image, and grabbed 4 images from the 360 view at angles 0,90,180 and 270 degrees.

![Pic of results5](https://github.com/kvsnoufal/ImageGeoLocation/blob/main/docs/5.JPG)


## Evaluation

Method: Group KFold - 10 splits - grouped by location

![Pic of results6](https://github.com/kvsnoufal/ImageGeoLocation/blob/main/docs/6.JPG)

### Confusion Matrix
![Pic of results7](https://github.com/kvsnoufal/ImageGeoLocation/blob/main/docs/7.JPG)

Average Accuracy: 25%


## Shoulders of giants
1. PlaNet geolocation with Convolutional Neural Networks - https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/45488.pdf
2. DeepGeo: Photo Localization with Deep Neural Network - https://arxiv.org/abs/1810.03077
3. GradCam on ResNext: https://www.kaggle.com/skylord/grad-cam-on-resnext
