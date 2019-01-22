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
2. We should see something like this:
```
Supraptos-MacBook-Pro:vgg16 suprapto$ 2019-01-22 02:45:08.971417: I tensorflow_serving/model_servers/server.cc:82] Building single TensorFlow model file config:  model_name: vgg16 model_base_path: /models/vgg16
2019-01-22 02:45:08.975002: I tensorflow_serving/model_servers/server_core.cc:461] Adding/updating models.
2019-01-22 02:45:08.975072: I tensorflow_serving/model_servers/server_core.cc:558]  (Re-)adding model: vgg16
2019-01-22 02:45:09.100691: I tensorflow_serving/core/basic_manager.cc:739] Successfully reserved resources to load servable {name: vgg16 version: 1}
2019-01-22 02:45:09.100787: I tensorflow_serving/core/loader_harness.cc:66] Approving load for servable version {name: vgg16 version: 1}
2019-01-22 02:45:09.100833: I tensorflow_serving/core/loader_harness.cc:74] Loading servable version {name: vgg16 version: 1}
2019-01-22 02:45:09.101632: I external/org_tensorflow/tensorflow/contrib/session_bundle/bundle_shim.cc:363] Attempting to load native SavedModelBundle in bundle-shim from: /models/vgg16/1
2019-01-22 02:45:09.101727: I external/org_tensorflow/tensorflow/cc/saved_model/reader.cc:31] Reading SavedModel from: /models/vgg16/1
2019-01-22 02:45:09.118671: I external/org_tensorflow/tensorflow/cc/saved_model/reader.cc:54] Reading meta graph with tags { serve }
2019-01-22 02:45:09.129217: I external/org_tensorflow/tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2019-01-22 02:45:09.181768: I external/org_tensorflow/tensorflow/cc/saved_model/loader.cc:162] Restoring SavedModel bundle.
2019-01-22 02:45:09.479903: W external/org_tensorflow/tensorflow/core/framework/allocator.cc:122] Allocation of 67108864 exceeds 10% of system memory.
2019-01-22 02:45:10.094330: W external/org_tensorflow/tensorflow/core/framework/allocator.cc:122] Allocation of 67108864 exceeds 10% of system memory.
2019-01-22 02:45:10.224545: W external/org_tensorflow/tensorflow/core/framework/allocator.cc:122] Allocation of 411041792 exceeds 10% of system memory.
2019-01-22 02:45:12.809462: W external/org_tensorflow/tensorflow/core/framework/allocator.cc:122] Allocation of 411041792 exceeds 10% of system memory.
2019-01-22 02:45:13.144690: I external/org_tensorflow/tensorflow/cc/saved_model/loader.cc:138] Running MainOp with key legacy_init_op on SavedModel bundle.
2019-01-22 02:45:13.144792: I external/org_tensorflow/tensorflow/cc/saved_model/loader.cc:259] SavedModel load for tags { serve }; Status: success. Took 4043061 microseconds.
2019-01-22 02:45:13.145941: I tensorflow_serving/servables/tensorflow/saved_model_warmup.cc:83] No warmup data file found at /models/vgg16/1/assets.extra/tf_serving_warmup_requests
2019-01-22 02:45:13.159513: I tensorflow_serving/core/loader_harness.cc:86] Successfully loaded servable version {name: vgg16 version: 1}
2019-01-22 02:45:13.165160: I tensorflow_serving/model_servers/server.cc:286] Running gRPC ModelServer at 0.0.0.0:8500 ...
[warn] getaddrinfo: address family for nodename not supported
2019-01-22 02:45:13.171820: I tensorflow_serving/model_servers/server.cc:302] Exporting HTTP/REST API at:localhost:8501 ...
[evhttp_server.cc : 237] RAW: Entering the event loop ...
```
3. Check the Docker container using following command:
```
$ docker ps -a
```
4. We should see something like this:
```
CONTAINER ID        IMAGE                COMMAND                  CREATED             STATUS              PORTS                              NAMES
9cd40faaefbc        tensorflow/serving   "/usr/bin/tf_serving…"   28 seconds ago      Up 26 seconds       0.0.0.0:8500->8500/tcp, 8501/tcp   vgg16_tf-serving-model
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
