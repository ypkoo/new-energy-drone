import pandas as pd
import numpy as np
from pykalman import KalmanFilter
from math import sqrt, radians, sin, cos
import os.path
from sklearn.preprocessing import StandardScaler

def get_df(filename=None):

	df = pd.read_csv(filename)


	return df

def make_history_(data, index_list, history_num):

		history = pd.DataFrame()


		for i in index_list:
			if history_num > 0:
				for n in range(history_num):
					if not i+'_shifted_by_'+str(n+1) in data.columns:
						data[i+'_shifted_by_'+str(n+1)] = data[i].shift(n+1)
			else:
				for n in range(-history_num):
					if not i+'_shifted_by_'+str(-(n+1)) in data.columns:
						data[i+'_shifted_by_'+str(-(n+1))] = data[i].shift(-(n+1))

		if history_num > 0:
			return data[history_num:]
		else:
			return data[:history_num]
		# data.drop(data.index[])
		# df.drop(df.index[[1,3]], inplace=True)
		# return history

def make_history(data, index_list, history_num):

	for i in index_list:
		for n in range(history_num):
			data[i+'_'+str(n+1)] = data[i].shift(n+1)

	# print(data[history_num:])
	return data[history_num:]

def make_target_feature(data, index_list, shift_num):

	for i in index_list:
		data[i+'_target'] = data[i].shift(-shift_num)

	return data[:-shift_num]

def make_input_for_cnn(data, index_list, output_index, history_num):

	inputs = []

	for i in index_list:
		temp = []
		for n in range(data.shape[0] - history_num):
			# print (data[i][n:n+history_num-1].values.shape)
			temp.append(data[i][n:n+history_num].values)
		inputs.append(pd.DataFrame(temp))

	return inputs






def get_moving_average(data, index_list, window):

	for i in index_list:
		data[i] = data[i].rolling(window=window).mean()

	return data[window-1:]

def get_act_vel(data):

	data['act_v'] = (data['act_vx']**2 + data['act_vy']**2)**.5

	return data


def get_vel(data):
	data['vel'] = (data['vel_x']**2 + data['vel_y']**2 + data['vel_z']**2)**.5

	return data


def get_vel_xy(data):
	data['vel'] = (data['vel_x']**2 + data['vel_y']**2)**.5


def get_dot_product(data):

	data['dot_p'] = data['act_vx'] * data['vel_x'] + data['act_vy'] * data['vel_y'] + data['act_vz'] * data['vel_z']

	return data

def get_act_acc(data):

	data['act_acc'] = ((data['act_vx']-data['vel_x'])**2 + (data['act_vy']-data['vel_y'])**2 + (data['act_vz']-data['vel_z'])**2)**.5
	# data['act_acc'] = ((data['act_vx']-data['vel_x'])**2 + (data['act_vy']-data['vel_y'])**2)**.5

	return data

def get_acc(data):

	data['acc'] = (data['acc_x']**2 + data['acc_y']**2 + data['acc_z']**2)**.5

	acc_norm = StandardScaler().fit_transform(data['acc'])
	acc_norm = pd.DataFrame(data=acc_norm, columns=['acc_norm'])

	power_norm = StandardScaler().fit_transform(data['power'])
	power_norm = pd.DataFrame(data=power_norm, columns=['power_norm'])

	ret = np.concatenate([acc_norm, power_norm])


	return ret

def get_acc_xy(data):

	data['acc'] = (data['acc_x']**2 + data['acc_y']**2)**.5


def get_vel_and_acc(data):
	get_vel(data)
	get_acc(data)

def get_vel_and_acc_xy(data):
	get_vel_xy(data)
	get_acc_xy(data)

def concat(filelist):
	df_list = []

	for f in filelist:
		df = pd.read_csv(f)
		# df = df.drop(df[df.isnull().any(1)].index) # delete if a row contains NaN
		df_list.append(df)

	df_concat = pd.concat(df_list)

	return df_concat

def get_distance(data):
	get_vel(data)

	accum_dist = 0
	data.at[0, 'accum_dist'] = 0
	for i in range(1, len(data)):
		dist = (data.iloc[i]['timestamp'] - data.iloc[i-1]['timestamp'])*0.000001 * data.iloc[i]['vel']
		accum_dist = accum_dist + dist
		data.at[i, 'accum_dist'] = accum_dist
		data.at[i, 'distance'] = dist


def get_accum_power(data):

	accum_power = 0

	for i in range(0, len(data)):
		accum_power = accum_power + data.at[i, 'power'] 
		data.at[i, 'accum_power'] = accum_power


def get_accum_power_by_distance(data):

	STEP = 1
	ref_dist = STEP

	ret = []

	for i in range(0, len(data)):
		if data.at[i, 'accum_dist'] > ref_dist:
			ret.append(str(data.at[i, 'accum_power']))
			ref_dist = ref_dist + STEP

	return "\n".join(ret)

def delete_useless_power(data):

	index_list = []

	for n in range(len(data)-1):
		if data.at[n, 'power'] != data.at[n+1, 'power']:
			index_list.append(n+1)

	# n = (n+1)%10
	# data = data[n:len(data):10]
	data = data.iloc[index_list]
	# index = [i for i in range(n, len(data)-1, 10)]
	# print(n)
	# print(index)
	# data = data.iloc[index]

	return data



def delete_useless_power_shift(data, shift_num, history_num=0):

	index_list = []

	# data = shift(data, 'power', shift_num)
	data['power_target'] = data['power'].shift(-shift_num)
	data = data[:-shift_num]
	# print(data)
	for n in range(len(data)-(history_num+1)):
		if data.at[n+history_num, 'power_target'] != data.at[n+history_num+1, 'power_target']:
			index_list.append(n+history_num+1)

	# n = (n+1)%10
	# data = data[n:len(data):10]
	data = data.iloc[index_list]

	return data

def shift(data, shift_index, shift_num):
	data[shift_index] = data[shift_index].shift(-shift_num)

	return data[:-shift_num]

def remove_abnormal_data(data):
	data = data[(data.power >10500) & (data.power < 17000)]

	return data

def trim(data, lower_lim, upper_lim):
	data = data[(data.power_target > lower_lim) & (data.power_target < upper_lim)]

	return data 

def shuffle_power(data):

	data['power_shuffled'] = data['power'].transform(np.random.permutation)

	return data

def undersample(data, bin_size, sample_num, lower_lim, upper_lim, replace=False):

	sampled_data_list = []

	
	lower_lim = lower_lim
	while True:
		df = data[(data.power_target >= lower_lim) & (data.power_target < lower_lim + bin_size)]

		if replace == True:
			print("True")
			df = df.sample(sample_num, replace=True)
		else:
			if df.shape[0] > sample_num:
				df = df.sample(sample_num, replace=False)
			
		sampled_data_list.append(df)

		lower_lim = lower_lim + bin_size

		if lower_lim >= upper_lim:
			break

	return pd.concat(sampled_data_list)

# def concat_for_rnn(filelist):
# 	df_list = []

# 	for f in filelist:
# 		data = pd.read_csv(f)
# 		for n in range(len(data)-1):
# 			if data.at[n, 'power'] != data.at[n+1, 'power']:
# 				if n+1 < 10:
# 					data = data[n+1:]
# 					break

# 		for n in range(len(data)):
# 			if data.at[-(n+1), 'power'] != data.at[-(n+2), 'power']:
# 				if n+1 < 10:
# 					data = data[:-]

# 		# df = df.drop(df[df.isnull().any(1)].index) # delete if a row contains NaN
# 		df_list.append(df)

# 	df_concat = pd.concat(df_list)

# 	return df_concat

def reshape_for_rnn(data, time_window):

	x_labels = ['vel_x', 'vel_y', 'vel_z', 'acc_x', 'acc_y', 'acc_z', 'roll', 'pitch', 'yaw', 'act_vx', 'act_vy']

	rnn_data = []
	rnn_power = []

	for n in range(len(data)-1):
		if data.at[n, 'power'] != data.at[n+1, 'power']:
			if n+1 > time_window:
				rnn_data.append(data.iloc[n+1 - time_window : n+1][x_labels].values)
				# print(rnn_data[-1])
				# print(data.iloc[n+1 - 10 : n+1][x_labels].shape)
				rnn_power.append(data.iloc[n+1][['power']].values)

	

	rnn_data = np.array(rnn_data)
	rnn_power = np.array(rnn_power)

	# print(rnn_data.shape)
	# print(rnn_power.shape)

	# print(rnn_data[0])
	# print(rnn_power[0])

	return rnn_data, rnn_power

def rotate_features(df, angle):

	angle_rad = radians(angle)

	df_rotate = df.copy()

	# rotate velocity
	df_rotate['vel_x'] = df['vel_x']*cos(angle_rad) - df['vel_y']*sin(angle_rad)
	df_rotate['vel_y'] = df['vel_x']*sin(angle_rad) + df['vel_y']*cos(angle_rad)

	# rotate acceleration
	df_rotate['acc_x'] = df['acc_x']*cos(angle_rad) - df['acc_y']*sin(angle_rad)
	df_rotate['acc_y'] = df['acc_x']*sin(angle_rad) + df['acc_y']*cos(angle_rad)

	# rotate attitude

def trim_shift_concat(filelist):
	df_list = []

	for i in range(len(filelist)):
		df = pd.read_csv(filelist[i])
		df = delete_useless_power_shift(df, 10)
		# df = df.drop(df[df.isnull().any(1)].index) # delete if a row contains NaN
		df_list.append(df)

	df_concat = pd.concat(df_list)

	return df_concat


def count_decimal_places(data):
	for n in range(len(data)-1):
		power = str(data.at[n, 'power'])
		print(power[::-1].find('.'))


def save_csv(df, filename, postfix=None):
	base, ext = os.path.splitext(filename)

	if postfix:
		df.to_csv(base+'_'+postfix+ext, index=False, sep=",")
	else:
		df.to_csv(filename, index=False, sep=",")


