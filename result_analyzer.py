from tkinter import *
from tkinter.scrolledtext import ScrolledText
from tkinter.filedialog import askdirectory
from glob import glob
from abc import ABCMeta, abstractmethod
import pprint
import json
import subprocess
import os
import pandas as pd
from PIL import ImageTk, Image
import matplotlib.pyplot as plt

# class ResultElemStringFormatter(object):

# 	key_lookup_table = {
# 		TYPE: 'type',
# 		SHIFT_NUM: 'shift_num',
# 		HISTORY_NUM: 'history_num',
# 		TIME_WINDOW: 'time_window',
# 		HIT_RATIO
# 	}

# 	ELEM_NUM = 6

# 	TYPE, SHIFT_NUM, HISOTRY_NUM, TIME_WINDOW, HIT_RATIO, RMS = range(ELEM_NUM)

# 	def get_string(self):
		
# 		s = ""
# 		for i in range(ELEM_NUM):
# 			s = s + '{:10}'.format('test')


class ResultElemBase(object):

	def __init__(self, dir_):
		self.dir_name = dir_
		self.args = json.loads(open(dir_ + "/args.json").read())
		self.result = json.loads(open(dir_ + "/result.json").read())
		self.result['hit_ratio'] = 0.0
		self.result['rms'] = float(self.result['mse'])**0.5

		try:
			if self.args['act_vel_upper_threshold'] == None:
				self.args['act_vel_upper_threshold'] = '-'
			if self.args['act_vel_lower_threshold'] == None:
				self.args['act_vel_lower_threshold'] = '-'
			if self.args['train_data_size'] == None:
				self.args['train_data_size'] = '-'
		except:
			self.args['act_vel_upper_threshold'] = '-'
			self.args['act_vel_lower_threshold'] = '-'
			self.args['train_data_size'] = '-'

	@abstractmethod
	def get_info_string(self):
		pass

	def get_rms(self):
		return self.result['rms']

	def get_mse(self):
		return float(self.result['mse'])

	def get_hit_ratio(self):
		return self.result['hit_ratio']

	def get_acc(self):
		return self.result['acc']

	def compute_hit_ratio(self, threshold_percent):
		# df = pd.read_csv(self.dir_name+"pred_result.csv")
		# hit_count = 0
		# threshold = threshold_percent * 0.01
		# for i in range(len(df)):
		# 	if df.ix[i, 1] < df.ix[i, 0] + df.ix[i, 0]*threshold and df.ix[i, 1] > df.ix[i, 0] - df.ix[i, 0]*threshold:

		# 		hit_count = hit_count + 1

		df = pd.read_csv(self.dir_name+"result_with_features.csv")
		df_hit = df[abs(df.prediction-df.real) <= df.real*threshold_percent*0.01]

		self.result['hit_ratio'] = (len(df_hit) / len(df)) * 100

	def draw_hit_graph(self, threshold_percent):
		MARKER_SIZE = 4
		df = pd.read_csv(self.dir_name+"result_with_features.csv")
		df_hit = df[abs(df.prediction-df.real) <= df.real*threshold_percent*0.01]
		df_miss = df[abs(df.prediction-df.real) > df.real*threshold_percent*0.01]

		df_pred = pd.read_csv(self.dir_name+"pred_result.csv")
		df_pred.columns = ['real', 'prediction']
		df_pred['upper_line'] = df_pred['real']*(1+threshold_percent*0.01)
		df_pred['lower_line'] = df_pred['real']*(1-threshold_percent*0.01)

		print(df.shape, df_hit.shape, df_miss.shape)

		plt.rcParams["figure.figsize"] = [12, 5]

		plt.ylim(8500, 17500)

		plt.plot(df_pred['real'], 'o', markersize=3)
		plt.plot(df_pred['prediction'], 'o', markersize=3)
		
		plt.plot(df_pred['upper_line'], 'ro', markersize=1)
		plt.plot(df_pred['lower_line'], 'ro', markersize=1)
		# plt.plot(result_trend, 'r--')
		# plt.plot(result)
		plt.title('test result')
		plt.ylabel('power')
		# plt.xlabel('epoch')
		plt.legend(['real', 'prediction'], loc='upper left')
		# plt.show()
		plt.savefig(self.dir_name+'figures/test_result2.eps', format='eps', dpi=1200)
		plt.savefig(self.dir_name+'figures/test_result2.png')

		plt.gcf().clear()
		# plt.rcParams["figure.figsize"] = [10, 10]
		# # vel
		
		# plt.plot(df_hit.vel_x, df_hit.vel_y, 'go', markersize=MARKER_SIZE)
		# # plt.savefig(self.dir_name+'hitmap_vel_hit.png')
		# plt.plot(df_miss.vel_x, df_miss.vel_y, 'ro', markersize=MARKER_SIZE)

		# plt.title('hit map')
		# plt.ylabel('vel_y')
		# plt.xlabel('vel_x')
		# plt.legend(['hit', 'miss'], loc='upper left')
		# plt.savefig(self.dir_name+'hitmap_vel'+str(threshold_percent)+'.png')

		# plt.gcf().clear()

		# # acc
		# plt.plot(df_hit.acc_x, df_hit.acc_y, 'go', markersize=MARKER_SIZE)
		# plt.plot(df_miss.acc_x, df_miss.acc_y, 'ro', markersize=MARKER_SIZE)

		# plt.title('hit map')
		# plt.xlabel('acc_x')
		# plt.ylabel('acc_y')
		# plt.legend(['hit', 'miss'], loc='upper left')
		# plt.savefig(self.dir_name+'hitmap_acc'+str(threshold_percent)+'.png')

		# plt.gcf().clear()

		# # act
		# plt.plot(df_hit.act_vx, df_hit.act_vy, 'go', markersize=MARKER_SIZE)
		# plt.plot(df_miss.act_vx, df_miss.act_vy, 'ro', markersize=MARKER_SIZE)

		# plt.title('hit map')
		# plt.xlabel('act_vx')
		# plt.ylabel('act_vy')
		# plt.legend(['hit', 'miss'], loc='upper left')
		# plt.savefig(self.dir_name+'hitmap_act'+str(threshold_percent)+'.png')

		# plt.gcf().clear()

		# # roll
		# plt.plot(df_hit.roll, 'go', markersize=MARKER_SIZE)
		# plt.plot(df_miss.roll, 'ro', markersize=MARKER_SIZE)

		# plt.title('hit map')
		# plt.ylabel('roll')
		# plt.legend(['hit', 'miss'], loc='upper left')
		# plt.savefig(self.dir_name+'hitmap_roll'+str(threshold_percent)+'.png')

		# plt.gcf().clear()

		# # pitch
		# plt.plot(df_hit.pitch, 'go', markersize=MARKER_SIZE)
		# plt.plot(df_miss.pitch, 'ro', markersize=MARKER_SIZE)

		# plt.title('hit map')
		# plt.ylabel('pitch')
		# plt.legend(['hit', 'miss'], loc='upper left')
		# plt.savefig(self.dir_name+'hitmap_pitch'+str(threshold_percent)+'.png')

		# plt.gcf().clear()

		# # yaw
		# plt.plot(df_hit.yaw, 'go', markersize=MARKER_SIZE)
		# plt.plot(df_miss.yaw, 'ro', markersize=MARKER_SIZE)

		# plt.title('hit map')
		# plt.ylabel('yaw')
		# plt.legend(['hit', 'miss'], loc='upper left')
		# plt.savefig(self.dir_name+'hitmap_yaw'+str(threshold_percent)+'.png')

		# plt.gcf().clear()

class RNNElem(ResultElemBase):

	def get_info_string(self):

		s = '{:5}{:5}{:10}{:10}{:10}{:10}'.format(self.args['type'], 
													self.args['shift_num'], 
													'', 
													self.args['time_window'], 
													self.result['hit_ratio'], 
													self.result['rms'])

		return s


class FCElem(ResultElemBase):

	def get_info_string(self):
		s = '{:<5}{:<5}{:<5}{:<50}{:<5}{:<5}{:<5}{:<10.2f}{:<10.2f}'.format(self.args['type'], 
													self.args['shift_num'], 
													self.args['history_num'], 
													str(self.args['hidden_layer_weights']),
													self.args['act_vel_upper_threshold'],
													self.args['act_vel_lower_threshold'],
													self.args['train_data_size'],
													self.result['hit_ratio'], 
													self.result['rms'])

		return s

def result_elem_factory(dir_):

	args = json.loads(open(dir_ + "/args.json").read())

	if args['type'] == 'rnn':
		return RNNElem(dir_)
	elif args['type'] == 'fc':
		return FCElem(dir_)
	else:
		return None

def text_click_callback(event):
	print("hi")
	print(event.x, event.y)
	# an event to highlight a line when single click is done
	line_no = event.widget.index("@%s,%s linestart" % (event.x, event.y))
	#print(line_no)
	line_end = event.widget.index("%s lineend" % line_no)
	event.widget.tag_remove("highlight", 1.0, "end")
	event.widget.tag_add("highlight", line_no, line_end)
	event.widget.tag_configure("highlight", background="yellow")

class DataPreProcessingWindow(Frame):

	# Define settings upon initialization. Here you can specify
	def __init__(self, master=None):
		
		Frame.__init__(self, master)   

		#reference to the master widget, which is the tk window                 
		self.master = master

		self.info_list = []
		self.selected_var = IntVar()
		self.clicked_elem_num = -1
		#with that, we want to then run init_window, which doesn't yet exist
		self.init_window()

	#Creation of init_window
	def init_window(self):

		# changing the title of our master widget      
		self.master.title("GUI")

		self.button_frame = Frame(self.master)
		self.list_frame = Frame(self.master)
		self.detail_frame = Frame(self.master)

		self.button_frame.grid(row=0, column=0)
		self.list_frame.grid(row=0, column=1)
		self.detail_frame.grid(row=0, column=2)


		self.cut_top_n = Entry(self.button_frame)
		file_select_btn = Button(self.button_frame, text="Select Files", width=15, command=self.file_select_btn_clicked)
		sort_by_hit_ratio = Button(self.button_frame, text="sort by hit ratio", width=15, command=self.sort_by_hit_ratio_btn_clicked)
		sort_by_rms = Button(self.button_frame, text="sort by rms", width=15, command=self.sort_by_rms_btn_clicked)
		self.hit_ratio_threshold = Entry(self.button_frame)
		compute_hit_ratio_btn = Button(self.button_frame, text="compute threshod", width=15, command=self.compute_hit_ratio)
		draw_hit_graph_btn = Button(self.button_frame, text="draw hit graph", width=15, command=self.draw_hit_graph)

		self.cut_top_n.pack(anchor="n", fill=BOTH)
		self.cut_top_n.insert(0, '100')
		file_select_btn.pack(anchor="n", fill=BOTH)
		sort_by_hit_ratio.pack(anchor="n", fill=BOTH)
		sort_by_rms.pack(anchor="n", fill=BOTH)
		self.hit_ratio_threshold.pack(anchor="n", fill=BOTH)
		compute_hit_ratio_btn.pack(anchor="n", fill=BOTH)
		draw_hit_graph_btn.pack(anchor="n", fill=BOTH)
		# file_select_btn.grid(row=0, column=0, sticky=N)
		# sort_by_hit_ratio.grid(row=1, column=0, sticky=N)
		# sort_by_rms.grid(row=2, column=0, sticky=N)
		# open_dir.grid(row=3, column=0, sticky=N)
		# self.hit_ratio_threshold.grid(row=4, column=0)
		# compute_hit_ratio_btn.grid(row=5, column=0)

		header_label = Entry(self.list_frame)
		header_label.pack(anchor="n", fill=BOTH)
		header_label.insert(END, '{:<5}{:<5}{:<5}{:<20}{:<5}{:<5}{:<5}{:<10}{:<10}'.format('type', 'shift', 'history', 'network', 'upper', 'lower', 'size', 'hit', 'rms') + "\n")

		self.list_elem_frame = ScrolledText(self.list_frame, width=100, height=65)
		self.list_elem_frame.pack()

		self.list_elem_frame.bind("<Button-1>", self.list_elem_clicked)
		self.list_elem_frame.bind("<Double-Button-1>", self.list_elem_double_clicked)

		self.detail_text = ScrolledText(self.detail_frame, height=65)
		self.detail_text.pack()

		# img = PhotoImage(file="loss.png")
		# self.loss_panel = Label(self.detail_frame, image=img)
		# self.loss_panel.pack(side = "bottom", fill = "both", expand = "yes")
		# self.file_list_text = ScrolledText(self.list_frame, width=80)
		# self.file_list_text.grid(row=0, column=0)

	def file_select_btn_clicked(self):

		# clear old elements
		self.info_list = []
		self.list_elem_frame.delete(1.0, END)


		dir_ = askdirectory(initialdir="C:/Users/lanada/Desktop/new_energy_drone/result")
		dirs = glob(dir_ + "/*/")

		for d in dirs:
			e = result_elem_factory(d)
			self.info_list.append(e)

		self.info_list.sort(key=lambda x: x.get_mse())

		self.info_list = self.info_list[:int(self.cut_top_n.get())]
		for i in self.info_list:
			self.list_elem_frame.insert(END, i.get_info_string()+"\n")
			# rb = Radiobutton(self.list_elem_frame, indicatoron=0, text=self.info_list[i].get_info_string(), variable=self.selected_var, value=i, command=self.list_btn_clicked) 
			# rb.pack(anchor="w", fill=BOTH)

	def list_elem_clicked(self, event):
		self.detail_text.delete(1.0, END)

		# an event to highlight a line when single click is done
		line_no = event.widget.index("@%s,%s linestart" % (event.x, event.y))
		line_end = event.widget.index("%s lineend" % line_no)
		event.widget.tag_remove("highlight", 1.0, "end")
		event.widget.tag_add("highlight", line_no, line_end)
		event.widget.tag_configure("highlight", background="yellow")

		self.clicked_elem_num = int(line_no.split('.')[0]) - 1

		s_args = pprint.pformat(self.info_list[self.clicked_elem_num].args, indent=4) + '\n\n'
		s_results = pprint.pformat(self.info_list[self.clicked_elem_num].result, indent=4)
		# self.detail_text.insert(END, self.info_list[self.clicked_elem_num].dir_name)
		self.detail_text.insert(END, s_args)
		self.detail_text.insert(END, s_results)

		# img = ImageTk.PhotoImage(Image.open(self.info_list[self.clicked_elem_num].dir_name+"loss.png").convert("RGB"))
		# img = PhotoImage(file=self.info_list[self.clicked_elem_num].dir_name+"loss.png")
		# self.loss_panel.configure(image=img)

	def list_elem_double_clicked(self, event):
		line_no = event.widget.index("@%s,%s linestart" % (event.x, event.y))
		self.clicked_elem_num = int(line_no.split('.')[0]) - 1
		dir_name = self.info_list[self.clicked_elem_num].dir_name
		os.startfile(dir_name)

	def sort_by_hit_ratio_btn_clicked(self):
		self.list_elem_frame.delete(1.0, END)
		self.detail_text.delete(1.0, END)

		self.info_list.sort(key=lambda x: x.get_acc(), reverse=True)
		for i in self.info_list:
			self.list_elem_frame.insert(END, i.get_info_string()+"\n")

	def sort_by_rms_btn_clicked(self):
		self.list_elem_frame.delete(1.0, END)
		self.detail_text.delete(1.0, END)

		self.info_list.sort(key=lambda x: x.get_mse())
		for i in self.info_list:
			self.list_elem_frame.insert(END, i.get_info_string()+"\n")

	def compute_hit_ratio(self):
		self.list_elem_frame.delete(1.0, END)

		threshold = float(self.hit_ratio_threshold.get())
		for i in self.info_list:
			i.compute_hit_ratio(threshold)
			self.list_elem_frame.insert(END, i.get_info_string()+"\n")

	def draw_hit_graph(self):
		threshold = float(self.hit_ratio_threshold.get())
		for i in self.info_list:
			i.draw_hit_graph(threshold)

	def print_header(self):
		self.list_elem_frame.insert(END, '{:10}{:10}{:10}{:10}{:10}{:10}'.format('type', 'shift', 'history', 'window', 'acc', 'mse') + "\n", "bold")
if __name__ == "__main__":
	# root window created. Here, that would be the only window, but
	# you can later have windows within windows.
	root = Tk()

	# root.geometry("400x300")

	#creation of an instance
	app = DataPreProcessingWindow(root)

	#mainloop 
	root.mainloop() 