import numpy as np


def get_one_hot_index(vec):
    for i in range(len(vec)):
        if vec[i] == 1:
            return i


def split_data(*inputs, split=0.25):
    length = len(inputs[0])
    split_index = int(length * (1 - split))
    outputs = []

    for input in inputs:
        if not len(input) == length:
            raise Exception('Inputs of different length! (%d and %d)' % (length, len(input)))

        outputs.append(input[:split_index])
        outputs.append(input[split_index:])

    return outputs


def split_test(*inputs, split=0.25):
    return split_data(*inputs, split=split)


def split_val(*inputs, split=0.25):
    return split_data(*inputs, split=split)


def iterate_minibatches(*inputs, batchsize=128, shuffle=True):
    length = len(inputs[0])
    for input in inputs:
        assert len(input) == length

    if shuffle:
        indices = np.arange(length)
        np.random.shuffle(indices)

    for start_index in range(0, length - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_index:start_index + batchsize]
        else:
            excerpt = slice(start_index, start_index + batchsize)

        batch = []
        for input in inputs:
            batch.append(input[excerpt])

        yield batch
