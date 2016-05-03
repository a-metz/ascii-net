import numpy as np

import nnet

class TestNetXOR:
	def __init__(self):
		num_inputs = 2
		num_hidden = 3
		num_outputs = 1

		# initialize layers with bias inputs
		self.hidden_layer = nnet.Layer(num_hidden, num_inputs + 1)
		self.output_layer = nnet.Layer(num_outputs, num_hidden + 1)

	def train(self, input_, expected_output):
		output_error = expected_output - self.evaluate(input_)
		hidden_error = self.output_layer.back_propagate(output_error)
		# remove bias error from hidden_error vector
		hidden_error = hidden_error[:-1]
		self.hidden_layer.back_propagate(hidden_error)
		return np.mean(np.abs(output_error))

	def evaluate(self, input_):
		# append bias to input vector
		input_ = np.append(input_, 1)
		hidden = self.hidden_layer.feed_forward(input_, binary=True)
		# append bias to hidden vector
		hidden = np.append(hidden, 1)
		output = self.output_layer.feed_forward(hidden, binary=True)
		return output

def run_tests():
	np.random.seed(0)

	input_data = np.asarray([
		[0, 0],
		[0, 1],
		[1, 0],
		[1, 1]])

	expected_output_data = np.asarray([
		[0],
		[1],
		[1],
		[0]])

	test_net_xor = TestNetXOR()

	for n in range(10000):
		error_sum = 0
		for input_, expected_output in zip(input_data, expected_output_data):
			error_sum += test_net_xor.train(input_, expected_output)
		if n % 1000 == 0:
			print 'training error sum:', error_sum

	for input_, expected_output in zip(input_data, expected_output_data):
		output = test_net_xor.evaluate(input_)
		print 'evaluate with input:', input_, 'expected output:', expected_output, 'output:', output

if __name__ == "__main__":
	run_tests()
