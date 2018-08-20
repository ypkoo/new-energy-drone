from tkinter import *
from tkinter.scrolledtext import ScrolledText
from tkinter.filedialog import askdirectory
from glob import glob
from abc import ABCMeta, abstractmethod
import pprint
import json

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

try:
    import tkinter as tk
    import tkinter.ttk as ttk
except:
    import Tkinter as tk
    import ttk as ttk

class VerticalScrollFrame(ttk.Frame):
    """A ttk frame allowing vertical scrolling only.
    Use the '.interior' attribute to place widgets inside the scrollable frame.
    Adapted from https://gist.github.com/EugeneBakin/76c8f9bcec5b390e45df.
    Amendments:
    1. Original logic for configuring the interior frame and canvas
       scrollregion left canvas regions exposed (not suppose to) and allowed
       vertical scrolling even when canvas height is greater than the canvas
       required height, respectively. I have provided a new logic to
       resolve these issues.
    2. Provided options to configure the styles of the ttk widgets.
    3. Tested in Python 3.5.2 (default, Nov 23 2017, 16:37:01),
                 Python 2.7.12 (default, Dec  4 2017, 14:50:18) and
                 [GCC 5.4.0 20160609] on linux.
    Author: Sunbear
    Website: https://github.com/sunbearc22
    Created on: 2018-02-26
    Amended on: 2018-03-01 - corrected __configure_canvas_interiorframe() logic.  
    """

    
    def __init__(self, parent, *args, **options):
        """
        WIDGET-SPECIFIC OPTIONS:
           style, pri_background, sec_background, arrowcolor,
           mainborderwidth, interiorborderwidth, mainrelief, interiorrelief 
        """
        # Extract key and value from **options using Python3 "pop" function:
        #   pop(key[, default])
        style          = options.pop('style',ttk.Style())
        pri_background = options.pop('pri_background','light grey')
        sec_background = options.pop('sec_background','grey70')
        arrowcolor     = options.pop('arrowcolor','black')
        mainborderwidth     = options.pop('mainborderwidth', 0)
        interiorborderwidth = options.pop('interiorborderwidth', 0)
        mainrelief          = options.pop('mainrelief', 'flat')
        interiorrelief      = options.pop('interiorrelief', 'flat')

        ttk.Frame.__init__(self, parent, style='main.TFrame',
                           borderwidth=mainborderwidth, relief=mainrelief)

        self.__setStyle(style, pri_background, sec_background, arrowcolor)

        self.__createWidgets(mainborderwidth, interiorborderwidth,
                             mainrelief, interiorrelief,
                             pri_background)
        self.__setBindings()


    def __setStyle(self, style, pri_background, sec_background, arrowcolor):
        '''Setup stylenames of outer frame, interior frame and verticle
           scrollbar'''        
        style.configure('main.TFrame', background=pri_background)
        style.configure('interior.TFrame', background=pri_background)
        style.configure('canvas.Vertical.TScrollbar', background=pri_background,
                        troughcolor=sec_background, arrowcolor=arrowcolor)

        style.map('canvas.Vertical.TScrollbar',
            background=[('active',pri_background),('!active',pri_background)],
            arrowcolor=[('active',arrowcolor),('!active',arrowcolor)])


    def __createWidgets(self, mainborderwidth, interiorborderwidth,
                        mainrelief, interiorrelief, pri_background):
        '''Create widgets of the scroll frame.'''
        self.vscrollbar = ttk.Scrollbar(self, orient='vertical',
                                        style='canvas.Vertical.TScrollbar')
        self.vscrollbar.pack(side='right', fill='y', expand='false')
        self.canvas = tk.Canvas(self,
                                bd=0, #no border
                                highlightthickness=0, #no focus highlight
                                yscrollcommand=self.vscrollbar.set,#use self.vscrollbar
                                background=pri_background #improves resizing appearance
                                )
        self.canvas.pack(side='left', fill='both', expand='true')
        self.vscrollbar.config(command=self.canvas.yview)

        # reset the view
        self.canvas.xview_moveto(0)
        self.canvas.yview_moveto(0)

        # create a frame inside the canvas which will be scrolled with it
        self.interior = ttk.Frame(self.canvas,
                                  style='interior.TFrame',
                                  borderwidth=interiorborderwidth,
                                  relief=interiorrelief)
        self.interior_id = self.canvas.create_window(0, 0,
                                                     window=self.interior,
                                                     anchor='nw')


    def __setBindings(self):
        '''Activate binding to configure scroll frame widgets.'''
        self.canvas.bind('<Configure>',self.__configure_canvas_interiorframe)
        

    def __configure_canvas_interiorframe(self, event):
        '''Configure the interior frame size and the canvas scrollregion'''
        #Force the update of .winfo_width() and winfo_height()
        self.canvas.update_idletasks() 

        #Internal parameters 
        interiorReqHeight= self.interior.winfo_reqheight()
        canvasWidth    = self.canvas.winfo_width()
        canvasHeight   = self.canvas.winfo_height()

        #Set interior frame width to canvas current width
        self.canvas.itemconfigure(self.interior_id, width=canvasWidth)
        
        # Set interior frame height and canvas scrollregion
        if canvasHeight > interiorReqHeight:
            #print('canvasHeight > interiorReqHeight')
            self.canvas.itemconfigure(self.interior_id,  height=canvasHeight)
            self.canvas.config(scrollregion="0 0 {0} {1}".
                               format(canvasWidth, canvasHeight))
        else:
            #print('canvasHeight <= interiorReqHeight')
            self.canvas.itemconfigure(self.interior_id, height=interiorReqHeight)
            self.canvas.config(scrollregion="0 0 {0} {1}".
                               format(canvasWidth, interiorReqHeight))

class ResultElemBase(object):

	def __init__(self, dir_):

		self.args = json.loads(open(dir_ + "/args.json").read())
		self.result = json.loads(open(dir_ + "/result.json").read())

	@abstractmethod
	def get_info_string(self):
		pass

	def get_rms(self):
		return self.result['rms']

	def get_mse(self):
		return float(self.result['mse'])

	def get_hit_ratio(self):
		return self.result['hit_ratio']

	def get_accuracy(self):
		return self.result['accuracy']


class RNNElem(ResultElemBase):

	def get_info_string(self):

		s = '{:10}{:10}{:10}{:10}{:10}{:10}'.format(self.args['type'], 
													self.args['shift_num'], 
													'', 
													self.args['time_window'], 
													self.result['acc'], 
													self.result['mse'])

		return s


class FCElem(ResultElemBase):

	def get_info_string(self):

		s = '{:10}{:10}{:10}{:10}{:10.5}{:10.6}'.format(self.args['type'], 
													self.args['shift_num'], 
													self.args['history_num'], 
													'', 
													self.result['acc'], 
													self.result['mse'])

		return s

def result_elem_factory(dir_):

	args = json.loads(open(dir_ + "/args.json").read())

	if args['type'] == 'rnn':
		return RNNElem(dir_)
	elif args['type'] == 'fc':
		return FCElem(dir_)
	else:
		return None


class DataPreProcessingWindow(Frame):

	# Define settings upon initialization. Here you can specify
	def __init__(self, master=None):
		
		Frame.__init__(self, master)   

		#reference to the master widget, which is the tk window                 
		self.master = master

		self.info_list = []
		self.selected_var = IntVar()
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


		file_select_btn = Button(self.button_frame, text="Select Files", width=15, command=self.file_select_btn_clicked)
		file_select_btn.grid(row=0, column=0)

		header_label = Label(self.list_frame, text='{:10}{:10}{:10}{:10}{:10}{:10}'.format('type', 'shift', 'history', 'window', 'acc', 'mse'))
		header_label.grid(row=0, column=0, sticky=N)

		scrollbar = Scrollbar(list_frame)
		# scrollbar.pack( side = RIGHT, fill=Y )
		self.list_elem_frame = Frame(self.list_frame, yscrollcommand = scrollbar.set)
		self.list_elem_frame.grid(row=1, column=0)

		scrollbar.config( command = list_elem_frame.yview )

		self.detail_text = ScrolledText(self.detail_frame)
		self.detail_text.grid(row=0, column=0)
		# self.file_list_text = ScrolledText(self.list_frame, width=80)
		# self.file_list_text.grid(row=0, column=0)

	def file_select_btn_clicked(self):

		# clear old elements
		self.info_list = []

		slaves = self.list_elem_frame.pack_slaves()
		for s in slaves:
			print("hi")
			s.destroy()


		dir_ = askdirectory()

		dirs = glob(dir_ + "/*/")

		for d in dirs:
			
			e = result_elem_factory(d)
			self.info_list.append(e)

		self.info_list.sort(key=lambda x: x.get_mse())

		for i in range(len(self.info_list)):
			# self.file_list_text.insert(END, i.get_info_string()+'\n')
			rb = Radiobutton(self.list_elem_frame, indicatoron=0, text=self.info_list[i].get_info_string(), variable=self.selected_var, value=i, command=self.list_btn_clicked) 
			rb.pack(anchor="w", fill=BOTH)

	def list_btn_clicked(self):
		self.detail_text.delete(1.0, END)

		item_num = self.selected_var.get()

		s = pprint.pformat(self.info_list[item_num].args, indent=4)
		self.detail_text.insert(END, s)

if __name__ == "__main__":
	# root window created. Here, that would be the only window, but
	# you can later have windows within windows.
	root = Tk()

	# root.geometry("400x300")

	#creation of an instance
	app = DataPreProcessingWindow(root)

	#mainloop 
	root.mainloop() 