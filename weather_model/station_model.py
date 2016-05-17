from lasagne import init
from lasagne.layers import InputLayer, LSTMLayer, DropoutLayer, SliceLayer, DenseLayer
from lasagne.nonlinearities import tanh, softmax

from .data import gen_station_data

from .lib import split_val

from .util import Model, BayesHyperOptim


class StationModel(Model):

    def get_default_param_filename(self):

        return 'params/station_model.p'

    def load_hyperparams(self, hyperparams):

        self.num_hidden = int(hyperparams['num_hidden'])
        self.num_epochs = int(hyperparams['num_epochs'])
        self.batch_size = int(hyperparams['batch_size'])
        self.dropout_val = float(hyperparams['dropout_val'])
        self.learning_rate = float(hyperparams['learning_rate'])
        self.grad_clip = float(hyperparams['grad_clip'])
        self.l2_reg_weight = float(hyperparams['l2_reg_weight'])

    def load_default_hyperparams(self):

        self.num_hidden = 250
        self.num_epochs = 31
        self.batch_size = 365
        self.dropout_val = 0.0
        self.learning_rate = 0.055
        self.grad_clip = 927
        self.l2_reg_weight = 0.0007

    def create_lstm_stack(self, net):

        net = LSTMLayer(
                net, self.num_hidden,
                grad_clipping=self.grad_clip,
                nonlinearity=tanh)

        net = DropoutLayer(net, self.dropout_val)

        return net

    def create_model(self, input_spread, output_spread):

        net = InputLayer(shape=(None, None, input_spread))

        for _ in range(2):
            net = self.create_lstm_stack(net)

        net = SliceLayer(net, -1, 1)

        net = DenseLayer(
                net,
                num_units=output_spread,
                W=init.Normal(),
                nonlinearity=softmax)

        return net

    def get_supp_model_params(self, train_Xs, train_y, val_Xs, val_y):

        temp_spread = len(train_Xs[0][0, 0, :])

        supp_model_params = {}
        supp_model_params['input_spread'] = temp_spread
        supp_model_params['output_spread'] = temp_spread

        return supp_model_params

    def get_data(self):

        min_train_X, min_train_y, min_test_X, min_test_y, _, _, _, _ = gen_station_data()
        train_X, val_X, train_y, val_y = split_val(min_train_X, min_train_y)
        train_Xs, val_Xs = [train_X], [val_X]

        return train_Xs, val_Xs, train_y, val_y


def bayes_hyper_optim_station():

    model = StationModel()

    hp_ranges = {
            'num_hidden': (100, 1024),
            'num_epochs': (5, 100),
            'batch_size': (64, 512),
            'dropout_val': (0, 0.9),
            'learning_rate': (1e-5, 1e-1),
            'grad_clip': (50, 1000),
            'l2_reg_weight': (0, 1e-1)}

    optim = BayesHyperOptim()

    optim.maximize(model, hp_ranges)


def main():

    # train_default_station()

    bayes_hyper_optim_station()


if __name__ == '__main__':
    main()
