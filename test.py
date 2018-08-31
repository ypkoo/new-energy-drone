from main import *


x_labels = ['vel_x', 'vel_y', 'vel_z', 'acc_x', 'acc_y', 'acc_z', 'roll', 'pitch', 'yaw', 'act_vx', 'act_vy']
x_labels2 = ['vel_x', 'vel_y', 'vel_z', 'roll', 'pitch', 'yaw', 'act_vx', 'act_vy']
x_labels_list = [x_labels2]
history_labels = ['vel_x', 'vel_y', 'vel_z', 'acc_x', 'acc_y', 'acc_z', 'roll', 'pitch', 'yaw', 'act_vx', 'act_vy']
y_label = ['power']


profile_dict = {}
profile_dict['title'] = 'no-preprocess'
arg_list = []

for sn in [0]:
	for hn in [0]:
	# for x_labels in x_labels_list:
		for bs in [256]:
			# for tds in [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]:
			# for trim in [(10000, 16500), (10500, 16000), (11000, 15500), (11500, 15000), (12000, 14500)]:
			for i in range(1):
				args = {}
				args['type'] = 'fc'
				args['shift_num'] = sn
				args['history_num'] = hn
				args['history_labels'] = history_labels[:]
				args['x_labels'] = x_labels[:]
				args['y_label'] = ['power']

				args['epochs'] = 3000
				args['batch_size'] = bs
				args['min_delta'] = 500
				args['early_stop'] = True

				args['file_list'] = None
				args['data_dir'] = 'feature-test'
				args['test_file_list'] = None
				# args['test_file_list'] = None
				args['train_data_size'] = 0.8

				# args['hidden_layer_weights'] = [width] * depth
				args['hidden_layer_weights'] = [100, 100, 100, 100]

				args['act_vel_upper_threshold'] = None
				args['act_vel_lower_threshold'] = None

				args['random_state'] = i

				# undersampling
				args['undersample'] = False
				args['lower_lim'] = 10000
				args['upper_lim'] = 17500
				args['replace'] = True
				args['bin_size'] = 100
				args['sample_num'] = 100
				arg_list.append(args)
		



profile_dict['arg_list'] = arg_list
learn(profile_dict)




# profile_dict2 = {}
# profile_dict2['title'] = 'threshold-test2'
# arg_list2 = []

# for sn in [25]:
# 	for hn in [2]:
# 		for ut in [3, 3.5, 4, 4.5, 5, 5.5]:
# 			args = {}
# 			args['type'] = 'fc'
# 			args['shift_num'] = sn
# 			args['history_num'] = hn
# 			args['history_labels'] = history_labels[:]
# 			args['x_labels'] = x_labels[:]
# 			args['y_label'] = y_label[:]

# 			args['epochs'] = 1000
# 			args['batch_size'] = 40

# 			args['file_list'] = None
# 			args['data_dir'] = '0821'
# 			args['test_file_list'] = None
# 			args['train_data_size'] = None

# 			# args['hidden_layer_weights'] = [width] * depth
# 			args['hidden_layer_weights'] = [100, 100, 100, 100]

# 			args['act_vel_upper_threshold'] = ut
# 			args['act_vel_lower_threshold'] = None
# 			arg_list2.append(args)

# for sn in [25]:
# 	for hn in [2]:
# 		for lt in [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]:
# 			args = {}
# 			args['type'] = 'fc'
# 			args['shift_num'] = sn
# 			args['history_num'] = hn
# 			args['history_labels'] = history_labels[:]
# 			args['x_labels'] = x_labels[:]
# 			args['y_label'] = y_label[:]

# 			args['epochs'] = 1000
# 			args['batch_size'] = 40

# 			args['file_list'] = None
# 			args['data_dir'] = '0821'
# 			args['test_file_list'] = None
# 			args['train_data_size'] = None

# 			# args['hidden_layer_weights'] = [width] * depth
# 			args['hidden_layer_weights'] = [100, 100, 100, 100]

# 			args['act_vel_upper_threshold'] = None
# 			args['act_vel_lower_threshold'] = lt
# 			arg_list2.append(args)

# profile_dict2['arg_list'] = arg_list2
# learn(profile_dict2)


# for sn in range(23, 30):
# 	for tw in range(1, 2):
# 		args = {}
# 		args['type'] = 'rnn'
# 		args['shift_num'] = sn
# 		args['time_window'] = tw
# 		args['x_labels'] = x_labels[:]
# 		args['y_label'] = y_label[:]

# 		args['epoch_num'] = 1000
# 		args['batch_size'] = 40
# 		arg_list.append(args)