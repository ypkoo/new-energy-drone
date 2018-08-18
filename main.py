#!/usr/bin/env python
#coding=utf8

import config
import log_result as log
import models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
import numpy as np
from keras import backend as K
import matplotlib.pyplot as plt
import pathlib
from keras.utils.vis_utils import plot_model
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
import json
import pprint
from glob import glob

import time

import input_preprocess as ip

FLAGS = config.flags.FLAGS


def learn(profile):

	sc = StandardScaler()

	title = profile['title']


	arg_list = profile['arg_list']

	for args in arg_list:
		pprint.pprint(args)
		

		type_ = args['type']

		shift_num = args['shift_num']
		x_labels = args['x_labels']
		y_label = args['y_label']

		# file_list = args['file_list']
		file_list = glob("data/"+args['data_dir']+"/*.csv")

		if type_ == 'rnn':

			time_window = args['time_window']

			now = time.localtime()
			s_time = "%02d%02d-%02d%02d%02d" % (now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)
			save_dir = "result/%s/rnn-sn%d-tw%d_%s/" % (title, shift_num, time_window, s_time)

			x_data_list = []
			y_data_list = []

			for f in file_list:
				df = pd.read_csv(f)
				df = ip.shift_power(df, shift_num)
				x_data = sc.fit_transform(df[x_labels])
				x_data = pd.DataFrame(data=x_data, columns=x_labels)
				df = x_data.join(df[y_label])

				x_data, y_data = ip.reshape_for_rnn(df, time_window=time_window)

				x_data_list.append(x_data)
				y_data_list.append(y_data)

			x_data = np.concatenate(x_data_list)
			y_data = np.concatenate(y_data_list)

			model = models.lstm(input_shape=(x_data.shape[1], x_data.shape[2]))

		elif type_ == 'fc':

			history_labels = args['history_labels']
			history_num = args['history_num']

			now = time.localtime()
			s_time = "%02d%02d-%02d%02d%02d" % (now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)
			save_dir = "result/%s/fc-sn%d-hn%d_%s/" % (title, shift_num, history_num, s_time)

			for l in history_labels:
				for n in range(history_num):
					x_labels.append(l+"_" + str(n+1))

			df_list = []

			for f in file_list:
				df = pd.read_csv(f)

				df = ip.make_history(df, history_labels, history_num)
				df = ip.delete_useless_power_shift(df, shift_num=shift_num, history_num=history_num)
				# df = df.drop(df[df.isnull().any(1)].index) # delete if a row contains NaN
				df_list.append(df)


			df_concat = pd.concat(df_list)



			x_data = sc.fit_transform(df_concat[x_labels])
			y_data = df_concat[y_label].values

			model = models.flexible_model_koo(input_dim=x_data.shape[1], output_dim=1)


		# make save directory
		pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)
		pathlib.Path(save_dir+"/weights").mkdir(parents=True, exist_ok=True)

		x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=FLAGS.test_size, random_state=FLAGS.seed)

		log.logger.info("x_shape: " + str(x_data.shape) + ", y_shape:" + str(y_data.shape))


		# callbacks
		checkpoint = ModelCheckpoint(filepath=save_dir+'weights.hdf5', save_best_only=True)
		earlystop = EarlyStopping(monitor='val_loss', patience=20, verbose=1)
		tensorboard = TensorBoard(log_dir=save_dir)

		# Start training
		history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=FLAGS.n_e,
				  batch_size=FLAGS.b_s, verbose=FLAGS.verbose, callbacks=[checkpoint, earlystop, tensorboard])

		# Save plots
		# print("figure size", plt.rcParams["figure.figsize"])
		plt.rcParams["figure.figsize"] = [12, 5]
		plt.plot(history.history['mean_acc'])
		plt.plot(history.history['val_mean_acc'])
		plt.title('model accuracy')
		plt.ylabel('accuracy')
		plt.xlabel('epoch')
		plt.legend(['train', 'test'], loc='upper left')
		# plt.show()
		plt.savefig(save_dir+'mean.eps', format='eps', dpi=1200)

		plt.gcf().clear()

		plt.ylim(0, 1300000)
		plt.plot(history.history['loss'])
		plt.plot(history.history['val_loss'])
		plt.title('model loss')
		plt.ylabel('loss')
		plt.xlabel('epoch')
		plt.legend(['train', 'test'], loc='upper left')
		# plt.show()
		plt.savefig(save_dir+'loss.eps', format='eps', dpi=1200)


		# load the best model
		model.load_weights(save_dir+'weights.hdf5')

		# Evaluate the model
		scores = model.evaluate(x_test, y_test)

		result = {}
		result['acc'] = str(scores[2] * 100)
		result['mse'] = str(str(scores[1]))

		with open(save_dir+'result.json', 'w') as json_file:
			json_file.write(json.dumps(result))

		### save the results ###

		# save prediction result
		predictions = model.predict(x_test)
		y_test_t = y_test.reshape((-1, 1))

		predictions_train = model.predict(x_train)
		y_train_t = y_train.reshape((-1, 1))

		result = np.concatenate((y_test_t,predictions),axis=1)
		result_train = np.concatenate((y_train_t, predictions_train), axis=1)

		np.savetxt(save_dir+"pred_result.csv", result, delimiter=",")
		np.savetxt(save_dir+"pred_result-train.csv", result_train, delimiter=",")


		# Save model
		model_json = model.to_json()
		with open(save_dir+"model.json", "w") as json_file:
			json_file.write(model_json)  # serialize model to JSON
		# model.save_weights(save_dir+"weight.h5")  # weight
		print("Save model ... done")

		# Save args
		with open(save_dir+'args.json', 'w') as json_file:
			json_file.write(json.dumps(args))

		# Save visualized model
		# plot_model(model, to_file=save_dir+'model_plot.png', show_shapes=True, show_layer_names=True)

		K.clear_session()
