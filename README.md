# Person-Detection
Person Detection with OpenCV and Python using YOLO

## Working Principle
To detect the object we have used YOLO. YOLO algorithm employs convolutional neural networks (CNN) to detect objects in real-time. As the name suggests, the algorithm requires only a single forward propagation through a neural network to detect objects. This means that prediction in the entire image is done in a single algorithm run.

## APPROACH
-Import configuration and weights of YOLO

-Start the video and pass each frame to YOLO 

-If the person is detected it will return its bounding box and confidence.

## Packages Used
1. Numpy
2. OpenCV
3. YOLO configuration and weights files = https://pjreddie.com/darknet/yolo/

## OS used
Windows 10
