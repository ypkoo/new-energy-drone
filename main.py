#!/usr/bin/env python
#coding=utf8

import models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
import numpy as np
from keras import backend as K
import matplotlib.pyplot as plt
import pathlib
from keras.utils.vis_utils import plot_model
from keras.utils import multi_gpu_model
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
import json
import pprint
from glob import glob

import time

import input_preprocess as ip


def reshape_data_rnn(args, file_list):
	sc = StandardScaler()

	shift_num = args['shift_num']
	x_labels = args['x_labels'][:]
	y_label = args['y_label']
	time_window = args['time_window']

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

	return x_data, y_data

def reshape_data_fc(args, file_list, save_dir):
	sc = StandardScaler()

	shift_num = args['shift_num']
	x_labels = args['x_labels']
	y_label = args['y_label']
	history_labels = args['history_labels']
	history_num = args['history_num']

	for l in history_labels:
		for n in range(history_num):
			args['x_labels'].append(l+"_" + str(n+1))

	df_list = []

	for f in file_list:
		df = pd.read_csv(f)

		df = ip.get_act_vel(df)
		

		df = ip.make_history(df, history_labels, history_num)
		if args['y_label'][0] == 'power':
			# df = ip.delete_useless_power_shift(df, shift_num=shift_num, history_num=history_num)
			df['power_target'] = df['power']
		else:
			df = ip.make_target_feature(df, y_label, shift_num)

		# df = df.drop(df[df.isnull().any(1)].index) # delete if a row contains NaN
		df_list.append(df)

	y_label = [y_label[0] + '_target']

	df_concat = pd.concat(df_list)
	df_concat.to_csv(save_dir+"raw_data.csv", index=False, sep=",")

	# if args['act_vel_upper_threshold'] != None:
	# 	df_concat = df_concat.drop(df_concat[df_concat.act_v > args['act_vel_upper_threshold']].index)
	# if args['act_vel_lower_threshold'] != None:
	# 	df_concat = df_concat.drop(df_concat[df_concat.act_v < args['act_vel_lower_threshold']].index)

	# df_concat = df_concat.drop(df_concat[df_concat.power > 16500].index)
	# df_concat = df_concat.drop(df_concat[df_concat.power < 10000].index)
	if args['y_label'][0] == 'power':
		df_concat = ip.trim(df_concat, args['lower_lim'], args['upper_lim'])

	if args['undersample'] == True:
		train, test = train_test_split(df_concat, test_size=0.2, random_state=args['random_state'])

		train = ip.undersample(train, args['bin_size'], args['sample_num'], args['lower_lim'], args['upper_lim'], replace=args['replace'])

		
	else:
		train, test = train_test_split(df_concat, test_size=1-args['train_data_size'], random_state=args['random_state'])

	train.to_csv(save_dir+"train_data.csv", index=False, sep=",")
	test.to_csv(save_dir+"test_data.csv", index=False, sep=",")
	# x_data = sc.fit_transform(df_concat[x_labels])
	x_train = train[args['x_labels']].values
	y_train = train[y_label].values

	x_test = test[args['x_labels']].values
	y_test = test[y_label].values

	return x_train, y_train, x_test, y_test


def learn(profile):

	# make a result directory
	pathlib.Path('result').mkdir(parents=True, exist_ok=True)

	title = profile['title']
	arg_list = profile['arg_list']

	for args in arg_list:
		pprint.pprint(args)
		
		type_ = args['type']

		shift_num = args['shift_num']
		x_labels = args['x_labels'][:]
		y_label = args['y_label']

		if args['file_list'] == None:
			file_list = glob("data/"+args['data_dir']+"/*.csv")
		else:
			file_list = args['file_list']

		if type_ == 'rnn':

			time_window = args['time_window']

			now = time.localtime()
			s_time = "%02d%02d-%02d%02d%02d" % (now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)
			save_dir = "result/%s/rnn-sn%d-tw%d_%s/" % (title, shift_num, time_window, s_time)

			x_train, y_train, x_test, y_test = reshape_data_rnn(args, file_list)

			

			model = models.lstm(input_shape=(x_data.shape[1], x_data.shape[2]))

		elif type_ == 'fc':

			history_labels = args['history_labels']
			history_num = args['history_num']

			now = time.localtime()
			s_time = "%02d%02d-%02d%02d%02d" % (now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)
			save_dir = "result/%s/fc-sn%d-hn%d_%s/" % (title, shift_num, history_num, s_time)

			# make save directory
			pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)

			x_train, y_train, x_test, y_test = reshape_data_fc(args, file_list, save_dir)

			# if args['test_file_list'] == None:
			# 	x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=1)
			# else:
			# 	# x_train = x_data
			# 	# y_train = y_data
			# 	x_train, _, y_train, _ = train_test_split(x_data, y_data, test_size=1-args['train_data_size'], random_state=1)
			# 	x_test, y_test = reshape_data_fc(args, args['test_file_list'], save_dir)

			model = models.flexible_model_koo(input_dim=x_train.shape[1], output_dim=1, weights=args['hidden_layer_weights'])

		

		# make save directory
		pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)
		pathlib.Path(save_dir+'figures/').mkdir(parents=True, exist_ok=True)

		# callbacks
		callbacks = []
		checkpoint = ModelCheckpoint(filepath=save_dir+'weights.hdf5', save_best_only=True)
		earlystop = EarlyStopping(monitor='val_loss', min_delta=args['min_delta'], patience=100, verbose=1)
		tensorboard = TensorBoard(log_dir=save_dir)

		callbacks.append(checkpoint)
		if args['early_stop'] == True:
			callbacks.append(earlystop)
		callbacks.append(tensorboard)

		# Start training
		# history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=args['epochs'],
		# 		  batch_size=args['batch_size'], verbose=1, callbacks=[checkpoint, earlystop, tensorboard])
		history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=args['epochs'],
				  batch_size=args['batch_size'], verbose=1, callbacks=callbacks)

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
		plt.savefig(save_dir+'figures/mean.eps', format='eps', dpi=1200)
		plt.savefig(save_dir+'figures/mean.png')

		plt.gcf().clear()


		# print(type(history.history['loss']))
		history_all = history.history['loss'] + history.history['val_loss']
		# plt.ylim(min(history_all), max(history_all))
		plt.ylim(0, 1500000)
		plt.plot(history.history['loss'])
		plt.plot(history.history['val_loss'])
		plt.title('model loss')
		plt.ylabel('loss')
		plt.xlabel('epoch')
		plt.legend(['train', 'test'], loc='upper left')
		# plt.show()
		plt.savefig(save_dir+'figures/loss.eps', format='eps', dpi=1200)
		plt.savefig(save_dir+'figures/loss.png')


		# load the best model
		model.load_weights(save_dir+'weights.hdf5')

		# Evaluate the model
		scores = model.evaluate(x_test, y_test)

		result = {}
		result['acc'] = str(scores[2] * 100)
		result['mse'] = str(scores[1])
		result['train_data_shape'] = str(x_train.shape)
		result['test_data_shape'] = str(x_test.shape)

		with open(save_dir+'result.json', 'w') as json_file:
			json_file.write(json.dumps(result))

		### save the results ###

		# save prediction result
		predictions = model.predict(x_test)
		y_test_t = y_test.reshape((-1, 1))

		print(x_test.shape, y_test.shape, predictions.shape)
		result_with_features = np.concatenate((x_test, y_test, predictions), axis=1)
		header = ",".join(args['x_labels']) + ",real,prediction"
		np.savetxt(save_dir+"result_with_features.csv", result_with_features, header=header, comments='', delimiter=",")

		predictions_train = model.predict(x_train)
		y_train_t = y_train.reshape((-1, 1))

		result = np.concatenate((y_test_t,predictions),axis=1)
		result_train = np.concatenate((y_train_t, predictions_train), axis=1)

		result = result[result[:,0].argsort()]
		result_train = result_train[result_train[:,0].argsort()]

		# result_trend = np.polyfit(np.arange(len(result)), result[:,1], 1)

		plt.gcf().clear()

		if args['y_label'][0] == 'power':
			plt.ylim(8500, 17500)
		else:
			plt.ylim(result[:,[0,1]].min(), result[:,[0,1]].max())
		plt.plot(result[:,0], 'o', markersize=3)
		plt.plot(result[:,1], 'o', markersize=3)
		# plt.plot(result_trend, 'r--')
		# plt.plot(result)
		plt.title('test result')
		plt.ylabel('power')
		# plt.xlabel('epoch')
		plt.legend(['real', 'prediction'], loc='upper left')
		# plt.show()
		plt.savefig(save_dir+'figures/test_result.eps', format='eps', dpi=1200)
		plt.savefig(save_dir+'figures/test_result.png')

		plt.gcf().clear()

		if args['y_label'][0] == 'power':
			plt.ylim(8500, 17500)
		else:
			plt.ylim(result_train[:,[0,1]].min(), result_train[:,[0,1]].max())
		plt.plot(result_train[:,0], 'o', markersize=3)
		plt.plot(result_train[:,1], 'o', markersize=3)
		# plt.plot(result)
		plt.title('train result')
		plt.ylabel('power')
		# plt.xlabel('epoch')
		plt.legend(['real', 'prediction'], loc='upper left')
		# plt.show()
		plt.savefig(save_dir+'figures/train_result.eps', format='eps', dpi=1200)
		plt.savefig(save_dir+'figures/train_result.png')

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
