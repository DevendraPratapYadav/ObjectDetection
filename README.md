# ObjectDetection

## Visit : [Project report and results](https://github.com/DevendraPratapYadav/ObjectDetection/blob/master/Report.pdf)          

Project : Computer Vision [CSL462] - IIT Ropar

Object detection is one of the classic Computer Vision problems. 
Many algorithms focusing on accuracy and speed have been proposed using traditional Computer Vision techniques as well as Machine Learning. A
direct application of object detection is autonomous vehicles. However, most of the datasets available are specific to western countries. We attempt to learn a object detection for autorickshaws, which are commonly found in India, but not western countries. The project implements a sliding window based algorithm to identify a bounding box for objects. We calculate the accuracy using Intersection over Union.


### Dataset used: [Autorikshaw Detection Challenge](http://cvit.iiit.ac.in/autorickshaw_detection)

### HOW TO RUN : 
Prerequisites: MATLAB
1) Place the dataset images in the 'images' folder in the project directory.
2) Train the classifier by executing :
```sh
   Train_ID.m
 ```
 2) Evaluate performance by executing :
 ```sh
   Test_ID.m
 ```
 
 NOTE: A total of 800 images are present in the dataset. First 600 are used for training and rest 200 are used for testing.
 
 Detected bounding boxes are displayed for each image along with IoU value.
 
 A pretrained neural net is provided as "myNet.mat".
