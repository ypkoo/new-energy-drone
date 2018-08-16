from main import *

profile_dict = {}

profile_dict['title'] = '0815'

arg_list = []

x_labels = ['vel_x', 'vel_y', 'vel_z', 'acc_x', 'acc_y', 'acc_z', 'roll', 'pitch', 'yaw', 'act_vx', 'act_vy']
history_labels = ['vel_x', 'vel_y', 'vel_z', 'acc_x', 'acc_y', 'acc_z', 'roll', 'pitch', 'yaw', 'act_vx', 'act_vy']
y_label = ['power']
file_list = ['2018-08-03.10_31_21random5-1.csv',
'2018-08-03.10_52_43random5-2.csv',
'2018-08-03.11_25_28random5-3.csv',
'2018-08-03.11_47_48random5-4.csv',
'2018-08-03.12_10_49random5-5.csv',
'2018-08-06.10_38_16random5-2sec-1.csv',
'2018-08-06.11_05_33random5-2sec-2.csv',
'2018-08-06.11_28_55random5-2sec-3.csv',
'2018-08-06.11_52_45random5-2sec-4.csv',
'2018-08-06.12_17_01random5-2sec-5.csv',
'2018-08-08.10_58_54testgpsfense.csv',
'2018-08-08.11_04_21random5-1sec-1.csv',
'2018-08-08.11_19_31random5-1sec-2.csv',
'2018-08-08.11_40_59random5-1sec-3.csv',
'2018-08-08.12_03_43random5-1sec-4.csv',
'2018-08-08.12_24_06random5-1sec-5.csv',
'2018-08-08.12_44_23random5-1sec-6.csv',
'2018-08-09.10_44_30random-vy5-2sec-1.csv',
'2018-08-09.11_14_54random-vy5-2sec-2.csv',
'2018-08-09.11_24_33randomvy5-2sec-4.csv',
'2018-08-09.11_40_24random-vy5-2sec-5.csv',
'2018-08-09.12_04_28random-vy5-2sec-6.csv',
'2018-08-09.12_26_46random-vy5-2sec-7.csv']



for sn in range(23, 30):
	for hn in range(5):
		args = {}
		args['type'] = 'fc'
		args['shift_num'] = sn
		args['history_num'] = hn
		args['history_labels'] = history_labels[:]
		args['x_labels'] = x_labels[:]
		args['y_label'] = y_label[:]
		args['file_list'] = file_list[:]
		arg_list.append(args)

for sn in range(23, 30):
	for tw in range(1, 2):
		args = {}
		args['type'] = 'rnn'
		args['shift_num'] = sn
		args['time_window'] = tw
		args['x_labels'] = x_labels[:]
		args['y_label'] = y_label[:]
		args['file_list'] = file_list[:]
		arg_list.append(args)

print("arg length" + str(len(arg_list)))

profile_dict['arg_list'] = arg_list

learn(profile_dict)