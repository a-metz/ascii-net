import numpy as np

import classic_mlp_model


def test_xor(runs=1):
    eval_errors = 0
    for seed in range(runs):
        print('random seed:', seed)
        np.random.seed(seed)

        inputs = np.asarray([[0, 0], [0, 1], [1, 0], [1, 1]])

        expected_outputs = np.asarray([[0], [1], [1], [0]])

        model_xor = classic_mlp_model.ClassicMLPModel(2, 2, 1)

        for n in range(1000):
            error = model_xor.train_batch(inputs,
                                          expected_outputs,
                                          learning_rate=10)
            if n % 100 == 0:
                print('training error:', error)

        for input_, expected_output in zip(inputs, expected_outputs):
            output = model_xor.evaluate(input_)
            eval_errors += np.abs(expected_output - np.around(output))
            print('eval input:', input_, 'expected output:', expected_output, 'output:', output)

        print()

    print('eval_errors:', eval_errors)
    assert eval_errors == 0


def test_random_binary(runs=1):
    eval_errors = 0
    for seed in range(runs):
        print('random seed:', seed)
        np.random.seed(seed)

        inputs = np.asarray([
            [0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1],
            [1, 1, 0], [1, 1, 1]
        ])

        # generate random training data
        expected_outputs = np.random.random_integers(0, 1, (8, 2))

        print('input data:')
        print(inputs)
        print('expected_outputs:')
        print(expected_outputs)

        model_random_binary = classic_mlp_model.ClassicMLPModel(3, 6, 2)

        for n in range(1000):
            error = model_random_binary.train_batch(inputs,
                                                    expected_outputs,
                                                    learning_rate=10)
            if n % 100 == 0:
                print('training error:', error)

        for input_, expected_output in zip(inputs, expected_outputs):
            output = model_random_binary.evaluate(input_)
            eval_errors += np.abs(expected_output - np.around(output))
            print('eval input:', input_, 'expected output:', expected_output, 'output:', output)

        print()

    print('eval_errors:', eval_errors)


if __name__ == "__main__":
    test_random_binary(runs=20)