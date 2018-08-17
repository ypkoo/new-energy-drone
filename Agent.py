from model import *

class AgentBase(object):

	def __init__(self):
		pass

	def fit(self):
		pass

	def evaluate(self):
		pass

	def predict(self):
		pass

	def save_results(self):
		pass

class FCAgent(AgentBase):

	def __init__(self, input_shape, output_shape, depth, hidden_node_size, param_list=None):

		self._model = FullyConnected(input_shape=input_shape, 
									output_shape=output_shape, 
									depth=depth, 
									hidden_node_size=hidden_node_size,
									param_list=param_list)

	def fit(self):
		self._model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=FLAGS.n_e, batch_size=FLAGS.b_s, verbose=FLAGS.verbose)

	def evaluate(x_train, y_train, validation_data=(x_test, y_test), epochs=FLAGS.n_e,
			  batch_size=FLAGS.b_s, verbose=FLAGS.verbose):
