#import libraries
from __future__ import print_function
import os
import sys

import tensorflow as tf
from keras import backend as K
from keras.applications.vgg16 import VGG16
from keras.models import Model

from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model.utils import build_tensor_info
from tensorflow.python.saved_model.signature_def_utils_impl import build_signature_def

def main(_):
	print("Start the training...")

	#import model
	model = VGG16(weights = 'imagenet', include_top = True, input_shape = (224, 224, 3))

	#get input layer
	input_layer = model.get_layer('input_1')
	input_layer_in = input_layer.input
	print("The input layer: ", input_layer_in)

	#get block5_pool layer
	block5_pool = model.get_layer('block5_pool')
	block5_pool_out = block5_pool.output
	print("The block5 pool layer: ", block5_pool_out)

	#get prediction layer
	predictions_layer = model.get_layer('predictions')
	predictions_layer_out = predictions_layer.output
	print("The predictions layer: ", predictions_layer_out)

	#create new model using additional output of bottleneck features
	vgg16_b_f_model = Model(inputs = input_layer_in, 
		outputs = [block5_pool_out, predictions_layer_out])

	#view the summary of the new model
	vgg16_b_f_model.summary()

	#get input layer
	b_f_input_layer = vgg16_b_f_model.get_layer('input_1')
	b_f_input_layer_in = b_f_input_layer.input
	print("The input layer: ", b_f_input_layer_in)

	#get output layer
	vgg16_b_f_outputs = vgg16_b_f_model.outputs
	print("The block5_pool layer: ", vgg16_b_f_outputs[0])
	print("The predictions layer: ", vgg16_b_f_outputs[1])

	#convert input layer, block5 pool layer, and predictions layer from tensor to tensor info
	input_layer_in_info = build_tensor_info(input_layer_in)
	block5_pool_out_info = build_tensor_info(block5_pool_out)
	predictions_layer_out_info = build_tensor_info(vgg16_b_f_outputs[1])
	
	#export model
	K.set_learning_phase(0)
	export_path_base = "./vgg16_bottleneck-features"
	model_version = "1"
	export_path = os.path.join(
		tf.compat.as_bytes(export_path_base),
		tf.compat.as_bytes(model_version))
	print("Exporting trained model to: ", export_path)

	prediction_signature = build_signature_def(
		inputs = {'images': input_layer_in_info},
		outputs = {'scores_1': block5_pool_out_info, 'scores_2': predictions_layer_out_info},
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