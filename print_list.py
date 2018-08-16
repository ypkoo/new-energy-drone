import sys
import pprint
import pickle
import json

with open ("result/" + sys.argv[1] + "/result_list_sorted", 'rb') as fp:
	result_list = pickle.load(fp)

	for r in result_list:
		f = open(r['dir_name'] + "args.json").read()
		args = json.loads(f)

		if args['type'] == 'rnn':
			print("type: %s, time_window: %d, shift_num: %d, rms: %f, hit_ratio: %f" % (args['type'], args['time_window'], args['shift_num'], r['rms'], r['hit_ratio']))
		elif args['type'] == 'fc':
			print("type: %s, history_num: %d, shift_num: %d, rms: %f, hit_ratio: %f" % (args['type'], args['history_num'], args['shift_num'], r['rms'], r['hit_ratio']))

# 	pprint.pprint(result_list)

# 	json_data=open(file_directory).read()

# data = json.loads(json_data)