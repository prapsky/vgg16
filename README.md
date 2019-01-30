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
2. We should see something like this:
```
Using TensorFlow backend.
Start the training...
2019-01-23 11:17:30.159896: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.1 instructions, but these are available on your machine and could speed up CPU computations.
2019-01-23 11:17:30.159923: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.2 instructions, but these are available on your machine and could speed up CPU computations.
2019-01-23 11:17:30.159933: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX instructions, but these are available on your machine and could speed up CPU computations.
2019-01-23 11:17:30.159941: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX2 instructions, but these are available on your machine and could speed up CPU computations.
2019-01-23 11:17:30.159948: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use FMA instructions, but these are available on your machine and could speed up CPU computations.
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         (None, 224, 224, 3)       0
_________________________________________________________________
block1_conv1 (Conv2D)        (None, 224, 224, 64)      1792
_________________________________________________________________
block1_conv2 (Conv2D)        (None, 224, 224, 64)      36928
_________________________________________________________________
block1_pool (MaxPooling2D)   (None, 112, 112, 64)      0
_________________________________________________________________
block2_conv1 (Conv2D)        (None, 112, 112, 128)     73856
_________________________________________________________________
block2_conv2 (Conv2D)        (None, 112, 112, 128)     147584
_________________________________________________________________
block2_pool (MaxPooling2D)   (None, 56, 56, 128)       0
_________________________________________________________________
block3_conv1 (Conv2D)        (None, 56, 56, 256)       295168
_________________________________________________________________
block3_conv2 (Conv2D)        (None, 56, 56, 256)       590080
_________________________________________________________________
block3_conv3 (Conv2D)        (None, 56, 56, 256)       590080
_________________________________________________________________
block3_pool (MaxPooling2D)   (None, 28, 28, 256)       0
_________________________________________________________________
block4_conv1 (Conv2D)        (None, 28, 28, 512)       1180160
_________________________________________________________________
block4_conv2 (Conv2D)        (None, 28, 28, 512)       2359808
_________________________________________________________________
block4_conv3 (Conv2D)        (None, 28, 28, 512)       2359808
_________________________________________________________________
block4_pool (MaxPooling2D)   (None, 14, 14, 512)       0
_________________________________________________________________
block5_conv1 (Conv2D)        (None, 14, 14, 512)       2359808
_________________________________________________________________
block5_conv2 (Conv2D)        (None, 14, 14, 512)       2359808
_________________________________________________________________
block5_conv3 (Conv2D)        (None, 14, 14, 512)       2359808
_________________________________________________________________
block5_pool (MaxPooling2D)   (None, 7, 7, 512)         0
_________________________________________________________________
flatten (Flatten)            (None, 25088)             0
_________________________________________________________________
fc1 (Dense)                  (None, 4096)              102764544
_________________________________________________________________
fc2 (Dense)                  (None, 4096)              16781312
_________________________________________________________________
predictions (Dense)          (None, 1000)              4097000
=================================================================
Total params: 138,357,544
Trainable params: 138,357,544
Non-trainable params: 0
_________________________________________________________________
The input:  [<tf.Tensor 'input_1:0' shape=(?, 224, 224, 3) dtype=float32>]
The output:  [<tf.Tensor 'predictions/Softmax:0' shape=(?, 1000) dtype=float32>]
The label is:  19
Exporting trained model to:  b'./vgg16/1'
Done exporting!
```

### Run ModelServer Container
1. Run ModelServer container using following command:
```
$ docker run -p 8500:8500 \
--name vgg16_tf-serving-model \
--mount type=bind,source=/Users/your_computer_name/vgg16/vgg16,target=/models/vgg16 \
-e MODEL_NAME=vgg16 \
-e EXPOSE_PORT_GRPC=8500 \
-e EXPOSE_PORT_REST=8501 \
-t tensorflow/serving &
```
2. We should see something like this:
```
2019-01-23 04:40:03.441098: I tensorflow_serving/model_servers/server.cc:82] Building single TensorFlow model file config:  model_name: vgg16 model_base_path: /models/vgg16
2019-01-23 04:40:03.441625: I tensorflow_serving/model_servers/server_core.cc:461] Adding/updating models.
2019-01-23 04:40:03.441791: I tensorflow_serving/model_servers/server_core.cc:558]  (Re-)adding model: vgg16
2019-01-23 04:40:03.567488: I tensorflow_serving/core/basic_manager.cc:739] Successfully reserved resources to load servable {name: vgg16 version: 1}
2019-01-23 04:40:03.567567: I tensorflow_serving/core/loader_harness.cc:66] Approving load for servable version {name: vgg16 version: 1}
2019-01-23 04:40:03.567630: I tensorflow_serving/core/loader_harness.cc:74] Loading servable version {name: vgg16 version: 1}
2019-01-23 04:40:03.569498: I external/org_tensorflow/tensorflow/contrib/session_bundle/bundle_shim.cc:363] Attempting to load native SavedModelBundle in bundle-shim from: /models/vgg16/1
2019-01-23 04:40:03.569621: I external/org_tensorflow/tensorflow/cc/saved_model/reader.cc:31] Reading SavedModel from: /models/vgg16/1
2019-01-23 04:40:03.584008: I external/org_tensorflow/tensorflow/cc/saved_model/reader.cc:54] Reading meta graph with tags { serve }
2019-01-23 04:40:03.590481: I external/org_tensorflow/tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2019-01-23 04:40:03.631637: I external/org_tensorflow/tensorflow/cc/saved_model/loader.cc:162] Restoring SavedModel bundle.
2019-01-23 04:40:03.963359: W external/org_tensorflow/tensorflow/core/framework/allocator.cc:122] Allocation of 67108864 exceeds 10% of system memory.
2019-01-23 04:40:04.529656: W external/org_tensorflow/tensorflow/core/framework/allocator.cc:122] Allocation of 67108864 exceeds 10% of system memory.
2019-01-23 04:40:04.696500: W external/org_tensorflow/tensorflow/core/framework/allocator.cc:122] Allocation of 411041792 exceeds 10% of system memory.
2019-01-23 04:40:08.107654: W external/org_tensorflow/tensorflow/core/framework/allocator.cc:122] Allocation of 411041792 exceeds 10% of system memory.
2019-01-23 04:40:08.556439: I external/org_tensorflow/tensorflow/cc/saved_model/loader.cc:138] Running MainOp with key legacy_init_op on SavedModel bundle.
2019-01-23 04:40:08.556626: I external/org_tensorflow/tensorflow/cc/saved_model/loader.cc:259] SavedModel load for tags { serve }; Status: success. Took 4986985 microseconds.
2019-01-23 04:40:08.562207: I tensorflow_serving/servables/tensorflow/saved_model_warmup.cc:83] No warmup data file found at /models/vgg16/1/assets.extra/tf_serving_warmup_requests
2019-01-23 04:40:08.583299: I tensorflow_serving/core/loader_harness.cc:86] Successfully loaded servable version {name: vgg16 version: 1}
2019-01-23 04:40:08.590362: I tensorflow_serving/model_servers/server.cc:286] Running gRPC ModelServer at 0.0.0.0:8500 ...
[warn] getaddrinfo: address family for nodename not supported
2019-01-23 04:40:08.597064: I tensorflow_serving/model_servers/server.cc:302] Exporting HTTP/REST API at:localhost:8501 ...
[evhttp_server.cc : 237] RAW: Entering the event loop ...
```
3. Check the Docker container using following command:
```
$ docker ps -a
```
4. We should see something like this:
```
CONTAINER ID        IMAGE                COMMAND                  CREATED             STATUS              PORTS                              NAMES
cd11e60f76fb        tensorflow/serving   "/usr/bin/tf_serving…"   2 minutes ago       Up 2 minutes        0.0.0.0:8500->8500/tcp, 8501/tcp   vgg16_tf-serving-model
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

## Setup using Bottleneck Features
1. [Train and Export with Bottleneck Features](#train-and-export-with-bottleneck-features)
2. [Run ModelServer Container with Bottleneck Features](#run-modelserver-container-with-bottleneck-features)
3. [Call Prediction with Bottleneck Features](#call-prediction-with-bottleneck-features)

### Train and Export with Bottleneck Features
1. Train and export the model with Bottleneck Features to be served by TensorFlow using following command:
```
$ python vgg16_train_bottleneck_features.py
```
2. We should see something like this:
```
Using TensorFlow backend.
Start the training...
2019-01-30 11:28:11.820725: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.1 instructions, but these are available on your machine and could speed up CPU computations.
2019-01-30 11:28:11.820760: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.2 instructions, but these are available on your machine and could speed up CPU computations.
2019-01-30 11:28:11.820772: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX instructions, but these are available on your machine and could speed up CPU computations.
2019-01-30 11:28:11.820781: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX2 instructions, but these are available on your machine and could speed up CPU computations.
2019-01-30 11:28:11.820789: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use FMA instructions, but these are available on your machine and could speed up CPU computations.
The input layer:  Tensor("input_1:0", shape=(?, 224, 224, 3), dtype=float32)
The block 5 pool layer:  Tensor("block5_pool/MaxPool:0", shape=(?, 7, 7, 512), dtype=float32)
The predictions layer:  Tensor("predictions/Softmax:0", shape=(?, 1000), dtype=float32)
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         (None, 224, 224, 3)       0
_________________________________________________________________
block1_conv1 (Conv2D)        (None, 224, 224, 64)      1792
_________________________________________________________________
block1_conv2 (Conv2D)        (None, 224, 224, 64)      36928
_________________________________________________________________
block1_pool (MaxPooling2D)   (None, 112, 112, 64)      0
_________________________________________________________________
block2_conv1 (Conv2D)        (None, 112, 112, 128)     73856
_________________________________________________________________
block2_conv2 (Conv2D)        (None, 112, 112, 128)     147584
_________________________________________________________________
block2_pool (MaxPooling2D)   (None, 56, 56, 128)       0
_________________________________________________________________
block3_conv1 (Conv2D)        (None, 56, 56, 256)       295168
_________________________________________________________________
block3_conv2 (Conv2D)        (None, 56, 56, 256)       590080
_________________________________________________________________
block3_conv3 (Conv2D)        (None, 56, 56, 256)       590080
_________________________________________________________________
block3_pool (MaxPooling2D)   (None, 28, 28, 256)       0
_________________________________________________________________
block4_conv1 (Conv2D)        (None, 28, 28, 512)       1180160
_________________________________________________________________
block4_conv2 (Conv2D)        (None, 28, 28, 512)       2359808
_________________________________________________________________
block4_conv3 (Conv2D)        (None, 28, 28, 512)       2359808
_________________________________________________________________
block4_pool (MaxPooling2D)   (None, 14, 14, 512)       0
_________________________________________________________________
block5_conv1 (Conv2D)        (None, 14, 14, 512)       2359808
_________________________________________________________________
block5_conv2 (Conv2D)        (None, 14, 14, 512)       2359808
_________________________________________________________________
block5_conv3 (Conv2D)        (None, 14, 14, 512)       2359808
_________________________________________________________________
block5_pool (MaxPooling2D)   (None, 7, 7, 512)         0
_________________________________________________________________
flatten (Flatten)            (None, 25088)             0
_________________________________________________________________
fc1 (Dense)                  (None, 4096)              102764544
_________________________________________________________________
fc2 (Dense)                  (None, 4096)              16781312
_________________________________________________________________
predictions (Dense)          (None, 1000)              4097000
=================================================================
Total params: 138,357,544
Trainable params: 138,357,544
Non-trainable params: 0
_________________________________________________________________
The input layer:  Tensor("input_1:0", shape=(?, 224, 224, 3), dtype=float32)
The block5_pool layer:  Tensor("block5_pool/MaxPool:0", shape=(?, 7, 7, 512), dtype=float32)
The predictions layer:  Tensor("predictions/Softmax:0", shape=(?, 1000), dtype=float32)
Exporting trained model to:  b'./vgg16_bottleneck-features/1'
Done exporting!
```

### Run ModelServer Container with Bottleneck Features
1. Run ModelServer container using following command:
```
$ docker run -p 8500:8500 \
--name vgg16_tf-serving-model \
--mount type=bind,source=/Users/your_computer_name/vgg16/vgg16_bottleneck_features,target=/models/vgg16_bottleneck_features \
-e MODEL_NAME=vgg16_bottleneck_features \
-e EXPOSE_PORT_GRPC=8500 \
-e EXPOSE_PORT_REST=8501 \
-t tensorflow/serving &
```
2. We should see something like this:
```
2019-01-30 04:35:02.507708: I tensorflow_serving/model_servers/server.cc:82] Building single TensorFlow model file config:  model_name: vgg16_bottleneck_features model_base_path: /models/vgg16_bottleneck_features
2019-01-30 04:35:02.511478: I tensorflow_serving/model_servers/server_core.cc:461] Adding/updating models.
2019-01-30 04:35:02.511860: I tensorflow_serving/model_servers/server_core.cc:558]  (Re-)adding model: vgg16_bottleneck_features
2019-01-30 04:35:02.631587: I tensorflow_serving/core/basic_manager.cc:739] Successfully reserved resources to load servable {name: vgg16_bottleneck_features version: 1}
2019-01-30 04:35:02.631665: I tensorflow_serving/core/loader_harness.cc:66] Approving load for servable version {name: vgg16_bottleneck_features version: 1}
2019-01-30 04:35:02.631690: I tensorflow_serving/core/loader_harness.cc:74] Loading servable version {name: vgg16_bottleneck_features version: 1}
2019-01-30 04:35:02.632159: I external/org_tensorflow/tensorflow/contrib/session_bundle/bundle_shim.cc:363] Attempting to load native SavedModelBundle in bundle-shim from: /models/vgg16_bottleneck_features/1
2019-01-30 04:35:02.632246: I external/org_tensorflow/tensorflow/cc/saved_model/reader.cc:31] Reading SavedModel from: /models/vgg16_bottleneck_features/1
2019-01-30 04:35:02.645928: I external/org_tensorflow/tensorflow/cc/saved_model/reader.cc:54] Reading meta graph with tags { serve }
2019-01-30 04:35:02.655583: I external/org_tensorflow/tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2019-01-30 04:35:02.704330: I external/org_tensorflow/tensorflow/cc/saved_model/loader.cc:162] Restoring SavedModel bundle.
2019-01-30 04:35:03.067297: W external/org_tensorflow/tensorflow/core/framework/allocator.cc:122] Allocation of 67108864 exceeds 10% of system memory.
2019-01-30 04:35:04.259973: W external/org_tensorflow/tensorflow/core/framework/allocator.cc:122] Allocation of 67108864 exceeds 10% of system memory.
2019-01-30 04:35:04.552370: W external/org_tensorflow/tensorflow/core/framework/allocator.cc:122] Allocation of 411041792 exceeds 10% of system memory.
2019-01-30 04:35:08.024038: W external/org_tensorflow/tensorflow/core/framework/allocator.cc:122] Allocation of 411041792 exceeds 10% of system memory.
2019-01-30 04:35:08.800867: I external/org_tensorflow/tensorflow/cc/saved_model/loader.cc:138] Running MainOp with key legacy_init_op on SavedModel bundle.
2019-01-30 04:35:08.800958: I external/org_tensorflow/tensorflow/cc/saved_model/loader.cc:259] SavedModel load for tags { serve }; Status: success. Took 6168706 microseconds.
2019-01-30 04:35:08.802236: I tensorflow_serving/servables/tensorflow/saved_model_warmup.cc:83] No warmup data file found at /models/vgg16_bottleneck_features/1/assets.extra/tf_serving_warmup_requests
2019-01-30 04:35:08.817511: I tensorflow_serving/core/loader_harness.cc:86] Successfully loaded servable version {name: vgg16_bottleneck_features version: 1}
2019-01-30 04:35:08.828125: I tensorflow_serving/model_servers/server.cc:286] Running gRPC ModelServer at 0.0.0.0:8500 ...
[warn] getaddrinfo: address family for nodename not supported
2019-01-30 04:35:08.841257: I tensorflow_serving/model_servers/server.cc:302] Exporting HTTP/REST API at:localhost:8501 ...
[evhttp_server.cc : 237] RAW: Entering the event loop ...
```
3. Check the Docker container using following command:
```
$ docker ps -a
```
4. We should see something like this:
```
CONTAINER ID        IMAGE                COMMAND                  CREATED             STATUS              PORTS                              NAMES
0bba70e4a300        tensorflow/serving   "/usr/bin/tf_serving…"   35 seconds ago      Up 33 seconds       0.0.0.0:8500->8500/tcp, 8501/tcp   vgg16_tf-serving-model
```

### Call Prediction with Bottleneck Features
1. Now we can call our model that is running in the Docker container using following command:
```
$ python vgg16_predict_bottleneck_features.py --image=./cat.jpg
```
2. We should see something like this:
```
Using TensorFlow backend.
The server hostport is:  localhost:8500
The image is:  ./cat.jpg
The result is:  [[('n03207941', 'dishwasher', 0.10127503424882889), ('n02123394', 'Persian_cat', 0.07714854925870895), ('n04553703', 'washbasin', 0.05456560105085373), ('n04554684', 'washer', 0.05264336243271828), ('n02105056', 'groenendael', 0.04825349897146225)]]
Time elapsed: 2.8194730281829834
```
