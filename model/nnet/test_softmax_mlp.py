import numpy as np

from .mlp import SoftmaxMLP


def test_random_binary(runs=1):
    eval_errors = 0
    for seed in range(runs):
        print('random seed:', seed)
        np.random.seed(seed)

        inputs = np.asarray([[0, 0], [0, 1], [1, 0], [1, 1]])

        # generate random training data
        expected_outputs = np.asarray([
            [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]
        ])

        print('input data:')
        print(inputs)
        print('expected_outputs:')
        print(expected_outputs)

        model_random_binary = SoftmaxMLP(2, 4, 4)

        for n in range(10000):
            error = model_random_binary.train_batch(inputs,
                                                    expected_outputs,
                                                    learning_rate=1)
            if n % 1000 == 0:
                print('training error:', error)

        for input_, expected_output in zip(inputs, expected_outputs):
            output = model_random_binary.evaluate(input_)
            eval_errors += np.abs(expected_output - np.around(output))
            print('eval input:', input_, 'expected output:', expected_output,
                  'output:', output)

        print()

    print('eval_errors:', eval_errors)


if __name__ == "__main__":
    test_random_binary(runs=10)