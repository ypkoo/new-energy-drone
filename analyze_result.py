from tkinter import *
from tkinter.scrolledtext import ScrolledText
from input_preprocess import *
from tkinter.filedialog import askopenfilenames
from glob import glob
import sys
from sklearn.metrics import mean_squared_error
import numpy as np
from operator import itemgetter
import pprint
import pickle

HIT_WINDOW = 0.07

def check_hit(row):
	if row.ix[:,1] < row.ix[:,0] + row.ix[:,0]*0.05 and row.ix[:,1] > row.ix[:,0] - row.ix[:,0]*0.05:
		return 1
	else:
		return 0

if __name__ == "__main__":
	# root window created. Here, that would be the only window, but
	# you can later have windows within windows.

	dirs = glob("result/" + sys.argv[1] + "/*/")

	result_list = []

	for d in dirs:
		try:
			df = pd.read_csv(d+"pred_result.csv")
			rms = np.sqrt(mean_squared_error(df.ix[:, 0], df.ix[:, 1]))


			hit_count = 0
			for i in range(len(df)):
				if df.ix[i, 1] < df.ix[i, 0] + df.ix[i, 0]*HIT_WINDOW and df.ix[i, 1] > df.ix[i, 0] - df.ix[i, 0]*HIT_WINDOW:
					hit_count = hit_count + 1

			# hit_result = df.apply(check_hit, axis=1)

			# hit_count = df.ix[:,0].sum(axis=0)

			item = {}
			item['dir_name'] = d
			item['rms'] = rms
			item['hit_ratio'] = (hit_count / len(df)) * 100
			result_list.append(item)
			print(rms)
		except:
			continue

	result_list_sorted = sorted(result_list, key=itemgetter('rms'))

	with open("result/" + sys.argv[1] + "/result_list_sorted", 'wb') as fp:
		pickle.dump(result_list_sorted, fp)

	# with open ('outfile', 'rb') as fp:
	# 	itemlist = pickle.load(fp)

	pprint.pprint(result_list_sorted)

	# root = Tk()

	# # root.geometry("400x300")

	# #creation of an instance
	# app = DataPreProcessingWindow(root)

	# #mainloop 
	# root.mainloop() 