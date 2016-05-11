import numpy as np

import nnet
import boolean_operator_model


def test_boolean_operator_model():
    eval_errors = 0
    for seed in range(20):
        print 'random seed:', seed
        np.random.seed(seed)

        inputs = np.asarray([[0, 0], [0, 1], [1, 0], [1, 1]])

        expected_outputs = np.asarray([[0], [1], [1], [0]])

        test_model_xor = boolean_operator_model.BooleanOperatorModel()

        for n in range(1000):
            error = test_model_xor.train_batch(inputs,
                                               expected_outputs,
                                               learning_rate=10)
            if n % 100 == 0:
                print 'training error:', error

        for input_, expected_output in zip(inputs, expected_outputs):
            output = test_model_xor.evaluate(input_)
            eval_errors += np.abs(expected_output - np.vectorize(
                lambda o: 0 if o < 0.5 else 1)(output))
            print 'eval input:', input_, 'expected output:', expected_output, 'output:', output

        print

    print 'eval_errors:', eval_errors


if __name__ == "__main__":
    test_boolean_operator_model()
