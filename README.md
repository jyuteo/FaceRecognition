# Face Recognition
Recognising faces of individuals among a group of people (with OpenCV and deep learning face recognition embeddings)
<br></br>

<p align="center"> 
   <img src="https://github.com/jyuteo/FaceRecognition/blob/master/test_output_overview.gif" width="750"/>
</p>

## Requirements
- Python
- OpenCV
- dlib
- face_recognition

  after installation of `dlib` with Python bindings, then install `face_recognition` using pip
  
  ```
  pip3 install face_recognition
  ```

## How it works?
The diagram *([source](https://www.sciencedirect.com/science/article/pii/S0167739X18331133?via%3Dihub))* shows the architecture of the face recognition system.

<p align="center"> 
   <img src="https://ars.els-cdn.com/content/image/1-s2.0-S0167739X18331133-gr2.jpg"/>
</p>

Instead of training a neural network classifier which required huge training dataset for each person, the recognition system is able to recognise faces from image or video input with small dataset of face image. This is through the **deep learning metric** method, with`dlib` and  `face_recognition` library. The [face_recognition module](https://github.com/ageitgey/face_recognition) has **pre-trained network** that can encode each face's fetaures and construct 128-d embeddings, *refer to the image ([source](https://medium.com/@ageitgey/machine-learning-is-fun-part-4-modern-face-recognition-with-deep-learning-c3cffc121d78))* below. We then compare the 128-d embeddings of input face to our labelled faces in training dataset, and **classify the face using k-NN algorithm**.

<p align="center"> 
   <img src="https://miro.medium.com/max/3254/1*6kMMqLt4UBCrN7HtqNHMKw.png" alt="drawing" width="600"/>
</p>

## What's in the Repository?

#### 1. [face_database_bp](https://github.com/jyuteo/FaceRecognition/tree/master/face_database_bp)

In this project, I decided to train the system to recognise faces of Blackpink members, Jennie, Jisoo, Lisa and Rose.
So, this folder contains 4 directories named by the members, with photos of their respective faces.

The structure of the dataset is as follow:
```
face_dataset_bp   
│
└───Jennie
│   └───Jennie1.png
│   └───Jennie2.png
│   └───...
└───Jisoo
│   └───Jisoo1.png
│   └───Jisoo2.png
│   └───...
└───Lisa
│   └───Lisa1.png
│   └───Lisa2.png
│   └───...
└───Rose
    └───Rose1.png
    └───Rose2.png
    └───...
```

#### 2. [face_unknown](https://github.com/jyuteo/FaceRecognition/tree/master/face_unknown)
This folder contains photos of each person which are not invloved in training the model. They are used to validate our model.

#### 3. [face_classifier.py](https://github.com/jyuteo/FaceRecognition/blob/master/face_classifier.py)
Receives a face image as input. \
`face_recognition.face_encodings` encodes the face image into 128-d embedddings.\
`face_recognition_knn.predict` compare the embeddings with the model that we trained and returns the predicted name.

#### 4. [face_recognition_knn.py](https://github.com/jyuteo/FaceRecognition/blob/master/face_recognition_knn.py)
`train` function trains the weight of the face recognition model from the `face_database` with k-NN algorithm.\
`predict`function will compare the encoded embeddings with the input encodings and predict the person which the face represents.

#### 5. [face_recognition_video](https://github.com/jyuteo/FaceRecognition/blob/master/face_recognition_video.py)
Perform face recognition from input video/ webcam.

#### 6. [trained_knn_model_bp_2.clf](https://github.com/jyuteo/FaceRecognition/blob/master/trained_knn_model_bp_2.clf)
An example of trained model that can be used directly.

#### 7. [test_video](https://www.youtube.com/watch?v=l3ORhQaMUR4)
The input video is not uploaded here due to the file size. Get it with this [link](https://www.youtube.com/watch?v=l3ORhQaMUR4).

## References
- Face recognition with OpenCV, Python, and deep learning \
https://www.pyimagesearch.com/2018/06/18/face-recognition-with-opencv-python-and-deep-learning/
- https://github.com/ageitgey/face_recognition
