# VGG16 Image Classification
VGG16 Image Classification using TensorFlow Serving and Docker container.

## Setup
1. [Train and Export](#train-and-export)
2. [Run ModelServer Container](#run-modelserver-container)
3. [Call Prediction](#call-prediction)

### Train and Export
1. Train and export the model to be served by TensorFlow using following command:
```
$ python vgg16_train.py
```

### Run ModelServer Container
1. Run ModelServer container using following command:
```
$ docker run -p 8500:8500 \
--name vgg16_tf-serving-model \
--mount type=bind,source=/Users/suprapto/my_tensorflow/vgg16/vgg16,target=/models/vgg16 -e MODEL_NAME=vgg16 \
-e EXPOSE_PORT_GRPC=8500 \
-e EXPOSE_PORT_REST=8501 \
-t tensorflow/serving &
```

### Call Prediction
1. Now we can call our model that is running in the Docker container using following command:
```
$ python vgg16_predict.py --image=./cat.jpg
```
2. We should see something like this:
```
Using TensorFlow backend.
The server hostport is:  localhost:8500
The image is:  ./cat.jpg
The result is:  [[('n03207941', 'dishwasher', 0.10127503424882889), ('n02123394', 'Persian_cat', 0.07714854925870895), ('n04553703', 'washbasin', 0.05456560105085373), ('n04554684', 'washer', 0.05264336243271828), ('n02105056', 'groenendael', 0.04825349897146225)]]
Time elapsed: 2.8194730281829834
```
