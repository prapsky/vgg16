#import libraries
from __future__ import print_function
import os
import sys

import tensorflow as tf
from keras import backend as K
from keras.applications.vgg16 import VGG16
from keras.optimizers import SGD
import numpy as np
import cv2

from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model.utils import build_tensor_info
from tensorflow.python.saved_model.signature_def_utils_impl import build_signature_def

def main(_):
	print("Start the training...")
	
	#import dataset
	model = VGG16(weights = 'imagenet', include_top = True)

	#test the model
	im = cv2.resize(cv2.imread('bird.jpg'), (224,224))
	im = np.expand_dims(im, axis=0)
	out = model.predict(im)
	print("The label is: ", np.argmax(out))

	#view the properties of the model
	model.summary()
	print("The input: ", model.inputs)
	print("The output: ", model.outputs)

	#get input layer
	input_layer = model.get_layer('input_1')
	input_layer_in = input_layer.input
	print("The input layer: ", input_layer_in)

	#using bottleneck features
	bottleneck_layer = model.get_layer('block5_pool')
	bottleneck_layer_out = bottleneck_layer.output
	print("The bottleneck layer: ", bottleneck_layer_out)

	#convert input layer and bottleneck layer from tensor to tensor info
	input_layer_in_info = build_tensor_info(input_layer_in)
	bottleneck_layer_out_info = build_tensor_info(bottleneck_layer_out)

	#export model
	K.set_learning_phase(0)
	export_path_base = "./vgg16"
	model_version = "1"
	export_path = os.path.join(
		tf.compat.as_bytes(export_path_base),
		tf.compat.as_bytes(model_version))
	print("Exporting trained model to: ", export_path)

	prediction_signature = build_signature_def(
		inputs = {'images': input_layer_in_info},
		outputs = {'scores': bottleneck_layer_out_info},
		method_name = signature_constants.PREDICT_METHOD_NAME)

	builder = saved_model_builder.SavedModelBuilder(export_path)
	with K.get_session() as sess:
		builder.add_meta_graph_and_variables(
			sess = sess,
			tags = [tag_constants.SERVING],
			signature_def_map = {'predict_images': prediction_signature})
		builder.save()
	
	#clear session	
	K.clear_session()
	print("Done exporting!")

if __name__ == '__main__':
	tf.app.run()