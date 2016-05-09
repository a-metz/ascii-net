import numpy as np

import nnet


class TestNetXOR:
	def __init__(self):
		num_inputs = 2
		num_hidden = 3
		num_outputs = 1

		# initialize layers with bias inputs and weight transform
		weight_transform = lambda w: -1 if w < 0 else 1

		self.hidden_layer = nnet.Layer(
			num_neurons=num_hidden,
			num_inputs=num_inputs + 1)

		self.output_layer = nnet.Layer(
			num_neurons=num_outputs,
			num_inputs=num_hidden + 1)

	def train(self, input_, expected_output, learning_rate):
		output_error = expected_output - self.evaluate(input_)
		hidden_error = self.output_layer.back_propagate(output_error, learning_rate=learning_rate)
		# remove bias error from hidden_error vector
		hidden_error = hidden_error[:-1]
		self.hidden_layer.back_propagate(hidden_error, learning_rate=learning_rate)
		return np.mean(np.abs(output_error))

	def evaluate(self, input_):
		# append bias to input vector
		input_ = np.append(input_, 1)
		hidden = self.hidden_layer.feed_forward(input_)
		# append bias to hidden vector
		hidden = np.append(hidden, 1)
		output = self.output_layer.feed_forward(hidden)
		return output

def run_test():
	eval_errors = 0
	for seed in range(20):
		print 'random seed:', seed
		np.random.seed(seed)

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

		for n in range(5000):
			random_row = np.random.random_integers(0, 3)
			input_ = input_data[random_row]
			expected_output = expected_output_data[random_row]
			error = test_net_xor.train(input_, expected_output, learning_rate=5)
			if n % 500 == 0:
				print 'training error:', error

		for input_, expected_output in zip(input_data, expected_output_data):
			output = test_net_xor.evaluate(input_)
			eval_errors += np.abs(expected_output - np.vectorize(lambda o: 0 if o < 0.5 else 1)(output))
			print 'eval input:', input_, 'expected output:', expected_output, 'output:', output
		
		print

	print 'eval_errors:', eval_errors

if __name__ == "__main__":
	run_test()
