# faceShield_detector


## Introduction
In difficult time like these where viruses like Covid 19 are spreading world-wide we need to keep safe distance from each other and wear appropriate safty gears this project looks into that topic it helps us recognize that a person is wearing faceshield or not so that overall safty of all is maintained.

## 🔭 &nbsp; <span style="color: #f2cf4a; font-family: Babas; font-size: 1.4em;">Functioning
* Input is taken from Live feed
* Mediapipe is the python library responsible for the detection of the presence of a person's face
* Face identifed is then processed for the pressence of a face shield
* The result of the detection is showed through text as well as on the detectoe screen.



Please note that this is still a work under progress and new ideas and contributions are welcome.
* Currently, the model is trained to detect face thorugh mediapipe and face shiled by facial points detection. I have plans to train the model for other Safty gears as well.
* Currently, only usb cameras are supported. Support for other cameras needs to be added.
* The tracker needs to be made robust.
* Integrate service (via mobile app or SMS) to send real-time notifications to supervisors present on the field.

 
 ## 🛠 &nbsp;<span style="color: #f2cf4a; font-family: Babas; font-size: 1.4em;">Tech Stack
</span>

* Python
* Mediapipe
* openCV
* numpy

## Training the model

### 1. Data preparation

**Data Collection**

The dataset containing images of people wearing mask, faceshield  and people with gloves were collected mostly from google search. Some images have been collected form kaggle . Download images for training from [train_image_folder](https://drive.google.com/drive/folders/1b5ocFK8Z_plni0JL4gVhs3383V7Q9EYH?usp=sharing). [kaggle](https://www.kaggle.com/sumansid/facemask-dataset),



**Organize the dataset into 3 folders:**
* train_image_folder <= the folder that contains the train images.
* test_image_folder <= the folder that contains the test images.
* valid_image_folder <= the folder that contains the validation images.
* all therse folder should contain there anotated fills

There is a one-to-one correspondence by file name between images and annotations. If the validation set is empty, the training set will be automatically splitted into the training set and validation set using the ratio of 0.8.


The model section defines the type of the model to construct as well as other parameters of the model such as the input image size and the list of anchors. The `labels` setting lists the labels to be trained on. Only images, which has labels being listed, are fed to the network. The rest images are simply ignored. By this way, a Dog Detector can easily be trained using VOC or COCO dataset by setting `labels` to `['dog']`.



 
 ### 3. Perform detection using trained model on live feed from webcam
 `main.py`
 The trained weighets can be used in local inviorm=nment as well as on google colab.


## Acknowledgements

* [Google/mediapipe](https://github.com/google/mediapipe) for training data.

