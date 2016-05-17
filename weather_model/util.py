from lasagne.updates import adagrad
from lasagne.layers import InputLayer
from lasagne.layers import get_output, get_all_params, get_all_param_values
from lasagne.regularization import regularize_layer_params, l2
from lasagne.layers.helper import get_all_layers

import theano
import theano.tensor as T

import pickle

from bayes_opt import BayesianOptimization

from .lib import iterate_minibatches


class Model():
    def __init__(self, hyperparams=None, param_filename=None):

        self.set_hyperparams(hyperparams)

        self.set_param_filename(param_filename)

    def set_param_filename(self, param_filename):

        if param_filename is not None:
            self.param_filename = param_filename

        else:
            self.param_filename = self.get_default_param_filename()

    def get_default_param_filename(self):

        return 'params/model.p'

    def set_hyperparams(self, hyperparams):

        if hyperparams is not None:
            self.load_hyperparams(hyperparams)

        else:
            self.load_default_hyperparams()

    def load_hyperparams(self, hyperparams):

        raise NotImplementedError()

    def load_default_hyperparams(self):

        raise NotImplementedError()

    def create_model(self, input_spread, output_spread):

        raise NotImplementedError()

    def get_supp_model_params(self, train_Xs, train_y, val_Xs, val_y):

        return None

    def build_train_loss(self, layers, train_output, target_values):

        l2_penalty = regularize_layer_params(layers, l2) * self.l2_reg_weight
        loss = T.nnet.categorical_crossentropy(train_output, target_values).mean()
        loss += l2_penalty

        return loss

    def build_test_loss(self, layers, test_output, target_values):

        test_loss = T.nnet.categorical_crossentropy(test_output, target_values).mean()

        return test_loss

    def build_test_acc(self, test_output, target_values):

        test_acc = T.mean(T.eq(
                T.argmax(test_output, axis=1),
                T.argmax(target_values, axis=1)),
            dtype=theano.config.floatX)

        return test_acc

    def build_train_updates(self, net, loss):

        all_params = get_all_params(net, trainable=True)
        updates = adagrad(loss, all_params, self.learning_rate)

        return updates

    def get_input_vars(self, layers):

        input_vars = []

        for layer in layers:
            if isinstance(layer, InputLayer):
                input_vars.append(layer.input_var)

        return input_vars

    def build_train_function(self, net, layers, input_vars):

        target_values = T.imatrix('target_output')

        train_output = get_output(net)

        loss = self.build_train_loss(layers, train_output, target_values)
        updates = self.build_train_updates(net, loss)

        train_fn = theano.function(
                input_vars + [target_values],
                loss, updates=updates, allow_input_downcast=True)

        return train_fn

    def build_test_function(self, net, layers, input_vars):

        target_values = T.imatrix('target_output')

        test_output = get_output(net, deterministic=True)

        test_loss = self.build_test_loss(layers, test_output, target_values)

        test_acc = self.build_test_acc(test_output, target_values)

        test_fn = theano.function(
                input_vars + [target_values],
                [test_loss, test_acc], allow_input_downcast=True)

        return test_fn

    def compute_train_metrics(self, train_fn, train_Xs, train_y):

        train_loss = 0
        train_batches = 0

        for batch in iterate_minibatches(*train_Xs, train_y):
            train_loss += train_fn(*batch)
            train_batches += 1

        train_loss /= train_batches

        return train_loss

    def compute_val_metrics(self, test_fn, val_Xs, val_y):

        val_loss = 0
        val_acc = 0
        val_batches = 0

        for batch in iterate_minibatches(*val_Xs, val_y):
            [loss, acc] = test_fn(*batch)
            val_loss += loss
            val_acc += acc
            val_batches += 1

        val_loss /= val_batches
        val_acc /= val_batches

        return val_loss, val_acc

    def display_train_metrics(self, train_metrics):

        print('Train loss: ' + str(train_metrics))

    def display_val_metrics(self, val_metrics):

        val_loss, val_acc = val_metrics

        print('Val Loss: ' + str(val_loss) + ' | Val Acc: ' + str(val_acc))

    def save_params(self, layers):

        params = get_all_param_values(layers)
        pickle.dump(params, open(self.param_filename, 'wb'))

    def perform_epoch(self, train_fn, test_fn,
            train_Xs, train_y, val_Xs, val_y,
            val=True, save=False, verbose=False):

        train_metrics = self.compute_train_metrics(train_fn, train_Xs, train_y)

        if verbose:
            self.display_train_metrics(train_metrics)

        if val:
            val_metrics = self.compute_val_metrics(test_fn, val_Xs, val_y)

            if verbose:
                self.display_val_metrics(val_metrics)

        return train_metrics, val_metrics

    def train_model(self, train_Xs, val_Xs, train_y, val_y,
            val=True,
            save=False,
            verbose=False):

        supp_model_params = self.get_supp_model_params(train_Xs, train_y, val_Xs, val_y)

        if verbose:
            print("Compiling functions ...")

        net = self.create_model(**supp_model_params)

        layers = get_all_layers(net)

        input_vars = self.get_input_vars(layers)

        train_fn = self.build_train_function(net, layers, input_vars)

        test_fn = self.build_test_function(net, layers, input_vars)

        for epoch in range(self.num_epochs):
            train_metrics, val_metrics = self.perform_epoch(
                    train_fn, test_fn, train_Xs, train_y, val_Xs, val_y,
                    val, save, verbose)

        if save:
            self.save_params(layers)

        if val:
            return val_metrics

    def get_data(self):

        raise NotImplementedError()

    def train_with_data(self):

        data = self.get_data()

        self.train_model(
                *data,
                save=True, verbose=True, val=True)

    def test_hyperparams(self, **hyperparams):

        self.set_hyperparams(hyperparams)

        data = self.get_data()

        val_loss, val_acc = self.train_model(
                *data,
                save=False, verbose=False, val=True)

        return val_acc


class HyperOptim():

    def get_results(self, optim):

        raise NotImplementedError()

    def display_results(self, results):

        print('Final Results: %f%%' % (results * 100))


class BayesHyperOptim(HyperOptim):

    def __init__(self,
            kappa=5, acq='ucb',
            corr='cubic', nugget=1,
            num_iter=50,
            num_init_points=50):

        self.kappa = kappa
        self.corr = corr
        self.nugget = nugget
        self.acq = acq
        self.num_iter = num_iter
        self.num_init_points = num_init_points

    def get_gp_params(self):

        gp_params = {
                'corr': self.corr,
                'nugget': self.nugget,
                'init_points': self.num_init_points,
                'n_iter': self.num_iter,
                'acq': self.acq,
                'kappa': self.kappa}

        return gp_params

    def get_results(self, optim):

        maximum = optim.res['max']['max_val']

        return maximum

    def maximize(self, model, hp_ranges):

        optim = BayesianOptimization(model.test_hyperparams, hp_ranges)

        gp_params = self.get_gp_params()

        optim.maximize(**gp_params)

        results = self.get_results(optim)

        self.display_results(results)
