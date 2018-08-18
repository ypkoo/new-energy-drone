from main import *

profile_dict = {}

profile_dict['title'] = '0819'

arg_list = []

x_labels = ['vel_x', 'vel_y', 'vel_z', 'acc_x', 'acc_y', 'acc_z', 'roll', 'pitch', 'yaw', 'act_vx', 'act_vy']
history_labels = ['vel_x', 'vel_y', 'vel_z', 'acc_x', 'acc_y', 'acc_z', 'roll', 'pitch', 'yaw', 'act_vx', 'act_vy']
y_label = ['power']




for sn in range(25, 35):
	for hn in range(5):
		for width in [10, 20, 30, 40, 50, 100, 150, 200, 250, 300]:
			for depth in range(10):
				args = {}
				args['type'] = 'fc'
				args['shift_num'] = sn
				args['history_num'] = hn
				args['history_labels'] = history_labels[:]
				args['x_labels'] = x_labels[:]
				args['y_label'] = y_label[:]

				args['epochs'] = 1000
				args['batch_size'] = 40

				args['file_list'] = None
				args['data_dir'] = 'callback_test'
				# args['test_file_list'] = ['data/callback_test/2018-08-08.12_44_23random5-1sec-6.csv']
				args['test_file_list'] = None

				args['hidden_layer_weights'] = [width] * depth

				args['action_threshold'] = 3
				arg_list.append(args)

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


profile_dict['arg_list'] = arg_list

learn(profile_dict)