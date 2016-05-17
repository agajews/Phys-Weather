import sys

from lasagne.updates import adagrad
from lasagne import init
from lasagne.layers import InputLayer, DropoutLayer, SliceLayer, DenseLayer
from lasagne.layers import LSTMLayer, CustomRecurrentLayer
from lasagne.layers import ConcatLayer
from lasagne.layers import Conv2DLayer, MaxPool2DLayer
from lasagne.layers import get_output, get_all_params, get_all_param_values
from lasagne.regularization import regularize_layer_params, l2
from lasagne.nonlinearities import tanh, softmax, rectify

import theano
import theano.tensor as T

import pickle

from bayes_opt import BayesianOptimization

from .data import gen_map_data

from .lib import iterate_minibatches, split_val


def create_cnn(dropout_val, embedding,
        num_filters, filter_size,
        pool_size,
        width, height, num_channels):

    l_in = InputLayer(shape=(None, num_channels, width, height))

    l_conv_1 = Conv2DLayer(
            l_in,
            num_filters=num_filters,
            filter_size=(filter_size, filter_size),
            nonlinearity=rectify,
            W=init.GlorotUniform())

    l_pool_1 = MaxPool2DLayer(
            l_conv_1,
            pool_size=(pool_size, pool_size))

    l_conv_2 = Conv2DLayer(
            l_pool_1,
            num_filters=num_filters,
            filter_size=(filter_size, filter_size),
            nonlinearity=rectify)

    l_pool_2 = MaxPool2DLayer(
            l_conv_2,
            pool_size=(pool_size, pool_size))

    l_forward_1 = DenseLayer(
            l_pool_2,
            num_units=embedding,
            nonlinearity=rectify)

    return l_forward_1


def create_model(dropout_val, num_hidden, grad_clip,
        num_filters, filter_size,
        pool_size,
        embedding,
        width, height, num_channels,
        timesteps,
        input_spread, output_spread):

    l_map_in = InputLayer(shape=(None, timesteps, num_channels, width, height))

    l_in_hid = create_cnn(
            dropout_val, embedding,
            num_filters, filter_size,
            pool_size,
            width, height, num_channels)

    l_hid_hid = DenseLayer(
            InputLayer(l_in_hid.output_shape),
            num_units=embedding)

    l_pre = CustomRecurrentLayer(
            l_map_in, l_in_hid, l_hid_hid)

    l_stat_in = InputLayer(shape=(None, timesteps, input_spread))

    l_conc = ConcatLayer([l_stat_in, l_pre], axis=2)

    l_forward_1 = LSTMLayer(
            l_conc, num_hidden,
            grad_clipping=grad_clip,
            nonlinearity=tanh)

    l_dropout_1 = DropoutLayer(l_forward_1, dropout_val)

    l_forward_2 = LSTMLayer(
            l_dropout_1, num_hidden,
            grad_clipping=grad_clip,
            nonlinearity=tanh)

    l_dropout_2 = DropoutLayer(l_forward_2, dropout_val)

    l_forward_slice = SliceLayer(l_dropout_2, -1, 1)

    l_out = DenseLayer(
            l_forward_slice,
            num_units=output_spread,
            W=init.Normal(),
            nonlinearity=softmax)

    net = l_out

    layers = [l_map_in, l_stat_in,
              l_in_hid, l_hid_hid,
              l_pre,
              l_forward_1, l_dropout_1,
              l_forward_2, l_dropout_2,
              l_forward_slice, l_out]

    return net, layers


def train_model(map_train_X, stat_train_X, train_y,
        map_val_X, stat_val_X, val_y,
        width,
        height,
        num_channels,
        timesteps,
        dropout_val=0.5,
        num_hidden=512,
        grad_clip=100,
        num_filters=32,
        filter_size=3,
        pool_size=3,
        embedding=50,
        num_epochs=50,
        batch_size=128,
        learning_rate=0.01,
        l2_reg_weight=0.1,
        val=True,
        save=False, save_filename='params/station_min.p',
        verbose=False):

    num_hidden = int(num_hidden)
    num_epochs = int(num_epochs)
    batch_size = int(batch_size)
    dropout_val = float(dropout_val)
    learning_rate = float(learning_rate)
    grad_clip = float(grad_clip)
    l2_reg_weight = float(l2_reg_weight)
    num_filters = int(num_filters)
    filter_size = int(filter_size)
    pool_size = int(pool_size)
    embedding = int(embedding)

    temp_spread = len(stat_train_X[0, 0, :])

    net, layers = create_model(
            dropout_val, num_hidden, grad_clip,
            num_filters, filter_size,
            pool_size,
            embedding,
            width, height, num_channels,
            timesteps,
            input_spread=temp_spread, output_spread=temp_spread)

    map_input_layer = layers[0]
    stat_input_layer = layers[1]

    l2_penalty = regularize_layer_params(layers, l2) * l2_reg_weight

    target_values = T.imatrix('target_output')
    network_output = get_output(net)

    loss = T.nnet.categorical_crossentropy(network_output, target_values).mean()
    loss += l2_penalty

    all_params = get_all_params(net, trainable=True)
    updates = adagrad(loss, all_params, learning_rate)

    test_output = get_output(net, deterministic=True)
    test_loss = T.nnet.categorical_crossentropy(test_output, target_values).mean()
    test_acc = T.mean(T.eq(
            T.argmax(test_output, axis=1),
            T.argmax(target_values, axis=1)),
        dtype=theano.config.floatX)

    if verbose:
        print("Compiling functions ...")

    train_fn = theano.function(
            [map_input_layer.input_var, stat_input_layer.input_var, target_values],
            loss, updates=updates, allow_input_downcast=True)
    test_fn = theano.function(
            [map_input_layer.input_var, stat_input_layer.input_var, target_values],
            [test_loss, test_acc], allow_input_downcast=True)

    for epoch in range(num_epochs):
        train_loss = 0
        train_batches = 0
        for batch in iterate_minibatches(map_train_X, stat_train_X, train_y):
            map_input, stat_input, targets = batch
            train_loss += train_fn(map_input, stat_input, targets)
            train_batches += 1

        train_loss /= train_batches

        if verbose:
            print('Epoch: ' + str(epoch) + ' | Loss: ' + str(train_loss))

        if val:
            val_loss = 0
            val_acc = 0
            val_batches = 0
            for batch in iterate_minibatches(map_val_X, stat_val_X, val_y):
                map_input, stat_input, targets = batch
                [loss, acc] = test_fn(map_input, stat_input, targets)
                val_loss += loss
                val_acc += acc
                val_batches += 1
            val_loss /= val_batches
            val_acc /= val_batches
            if verbose:
                print('Val Loss: ' + str(val_loss) + ' | Val Acc: ' + str(val_acc))

    if save:
        params = get_all_param_values(layers)
        pickle.dump(params, open(save_filename, 'wb'))

    if val:
        return val_loss, val_acc


def train_min(width=100, height=50, timesteps=10, color='rgb'):
    [min_stat_train_X, min_train_y,
     _, _, _, _, _, _,
     min_map_train_X, _] = gen_map_data(
             width=width, height=height,
             timesteps=timesteps, color=color)
    if color == 'rgb':
        num_channels = 3
    elif color == 'hsv':
        num_channels = 1

    [min_map_train_X, min_map_val_X,
     min_stat_train_X, min_stat_val_X,
     min_train_y, min_val_y] = split_val(min_map_train_X, min_stat_train_X, min_train_y)

    default_hp = {
        'dropout_val': 0.0,
        'num_hidden': 512,
        'grad_clip': 100,
        'num_filters': 32,
        'filter_size': 3,
        'pool_size': 3,
        'embedding': 50,
        'num_epochs': 50,
        'batch_size': 128,
        'learning_rate': 0.01,
        'l2_reg_weight': 0.0007}

    train_model(
            min_map_train_X, min_stat_train_X, min_train_y,
            min_map_val_X, min_stat_val_X, min_val_y,
            width=width, height=height, num_channels=num_channels,
            timesteps=timesteps,
            save=True, verbose=True, val=True,
            **default_hp)


def test_hyperparams(width, height, timesteps, **kwargs):
    width = int(width)
    height = int(height)
    timesteps = int(timesteps)

    [min_stat_train_X, min_train_y,
     _, _, _, _, _, _,
     min_map_train_X, _] = gen_map_data(
             width=width, height=height,
             timesteps=timesteps, color='hsv')

    [min_map_train_X, min_map_val_X,
     min_stat_train_X, min_stat_val_X,
     min_train_y, min_val_y] = split_val(min_map_train_X, min_stat_train_X, min_train_y)

    try:
        val_loss, val_acc = train_model(
                min_map_train_X, min_stat_train_X, min_train_y,
                min_map_val_X, min_stat_val_X, min_val_y,
                width=width, height=height, num_channels=1,
                timesteps=timesteps,
                save=False, verbose=False, val=True,
                **kwargs)
        return val_acc
    except KeyboardInterrupt:
        sys.exit()
    except Exception as e:
        print(e)
        return 0.0


def bayes_optim():
    hp_ranges = {
        'width': (20, 160),
        'height': (10, 70),
        'timesteps': (5, 15),
        'dropout_val': (0.0, 0.9),
        'num_hidden': (100, 512),
        'grad_clip': (50, 1000),
        'num_filters': (16, 48),
        'filter_size': (3, 3),
        'pool_size': (3, 3),
        'embedding': (1, 100),
        'num_epochs': (5, 100),
        'batch_size': (48, 512),
        'learning_rate': (1e-5, 1e-1),
        'l2_reg_weight': (0.0, 1e-1)}

    '''hp_ranges = {
        'width': (100, 100),
        'height': (50, 50),
        'timesteps': (10, 10),
        'dropout_val': (0.0, 0.0),
        'num_hidden': (512, 512),
        'grad_clip': (50, 1000),
        'num_filters': (32, 32),
        'filter_size': (3, 3),
        'pool_size': (3, 3),
        'embedding': (1, 50),
        'num_epochs': (5, 100),
        'batch_size': (48, 512),
        'learning_rate': (1e-5, 1e-1),
        'l2_reg_weight': (0.0, 1e-1)}'''
    optim = BayesianOptimization(test_hyperparams, hp_ranges)

    gp_params = {'corr': 'cubic',
                 'nugget': 1}
    optim.maximize(init_points=50, n_iter=50, acq='ucb', kappa=5, **gp_params)
    print('Final Results: %f%%' % (optim.res['max']['max_val'] * 100))


if __name__ == '__main__':
    # train_min(width=100, height=50, color='rgb')
    # train_min(width=100, height=50, color='hsv')
    bayes_optim()
