#Import libraries
from __future__ import print_function
import os
import sys

import tensorflow as tf
from keras import backend as K
from keras.applications.vgg16 import VGG16
from keras.optimizers import SGD
from keras.models import Model
import numpy as np
import cv2

from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model.signature_def_utils_impl import predict_signature_def

def main(_):
	print("Start the training...")
	
	#Import dataset
	model = VGG16(weights='imagenet', include_top=True)
	sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)

	#Compile the model
	model.compile(optimizer=sgd, loss='categorical_crossentropy')

	#View the properties of the model
	model.summary()
	print("The input: ", model.inputs)
	print("The output: ", model.outputs)

	#Test the model
	im = cv2.resize(cv2.imread('bird.jpg'), (224,224))
	im = np.expand_dims(im, axis=0)
	out = model.predict(im)
	print("The label is: ", np.argmax(out))

	#Export model
	K.set_learning_phase(0)
	export_path_base = "./vgg16"
	model_version = "1"
	export_path = os.path.join(
		tf.compat.as_bytes(export_path_base),
		tf.compat.as_bytes(model_version))
	print("Exporting trained model to: ", export_path)

	prediction_signature = predict_signature_def(
		inputs={'images': model.input},
		outputs={'scores': model.output})

	builder = saved_model_builder.SavedModelBuilder(export_path)
	with K.get_session() as sess:
		builder.add_meta_graph_and_variables(
			sess=sess,
			tags=[tag_constants.SERVING],
			signature_def_map={'predict_images': prediction_signature})
		builder.save()
	
	#Clear session	
	K.clear_session()
	print("Done exporting!")

if __name__ == '__main__':
	tf.app.run()