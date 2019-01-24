#Libraries
from __future__ import print_function
import sys
import time

import numpy as np 
import tensorflow as tf 

import grpc
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
from argparse import ArgumentParser

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions

def parse_args():
    parser = ArgumentParser(description='Request a TensorFlow server for a prediction on the image')
    parser.add_argument('-s', '--server',
                        dest='server',
                        default='localhost:8500',
                        help='prediction service host:port')
    parser.add_argument("-i", "--image",
                        dest="image",
                        default='',
                        help="path to image in JPEG format",)
    parser.add_argument('-p', '--image_path',
                        dest='image_path',
                        default='',
						help='path to images folder',)
    args = parser.parse_args()
    hostport = args.server
    return hostport, args.image, args.image_path == 'true'

def main(_):
	hostport, image, image_path = parse_args()
	print("The server hostport is: ", hostport)
	print("The image is: ", image)

	#start count time
	start = time.time()

	#preprocessing
	images_np = np.zeros((1, 224, 224, 3), dtype=np.float32)
	img = load_img(image, target_size=(224, 224))
	images_np[0] = img_to_array(img)
	images_np = preprocess_input(images_np)
	
	#construct grpc request to model server
	channel = grpc.insecure_channel(hostport)
	stub = prediction_service_pb2.PredictionServiceStub(channel)
	request = predict_pb2.PredictRequest()
	request.model_spec.name = 'vgg16'
	request.model_spec.signature_name = 'predict_images'
	request.inputs['images'].CopyFrom(tf.contrib.util.make_tensor_proto(images_np, shape=list(images_np.shape)))
	result = stub.Predict(request, 60.0)
	
	#get result and postprocess
	images_predictions_np = np.zeros((1, 1000), dtype=np.float32)
	images_predictions_np[0] = result.outputs['scores'].float_val
	images_predictions_np = images_predictions_np.astype(float)
	images_predictions_list = decode_predictions(images_predictions_np, top=5)
	print("The result is: ", images_predictions_list)

	#end count time
	end = time.time()
	time_diff = end - start
	print('Time elapsed: {}'.format(time_diff))

if __name__ == '__main__':
	tf.app.run()