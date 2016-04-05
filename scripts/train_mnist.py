#!/usr/bin/env python

from encdec_vae import util
from encdec_vae import train
from encdec_vae.nn.vae import Vae
from encdec_vae.nn.mnist_vae import MnistEncoder, MnistDecoder

import os
import logging

from chainer import cuda, Variable
import chainer.functions


def _create_variables(x_data):
    return [Variable(x_data)]


def _get_status(batch):
    x_data, label_x_data, t_data, label_t_data = batch
    return [
        ('size', x_data.shape[0])
    ]


def main(args):
    if args.gpu is not None:
        cuda.get_device(args.gpu).use()

    os.makedirs(args.model)

    # set up logger
    logger = logging.getLogger()
    logging.basicConfig(level=logging.INFO)
    log_path = os.path.join(args.model, 'log')
    file_handler = logging.FileHandler(log_path)
    fmt = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    # set up optimizer
    optimizer = util.list2optimizer(args.optim)

    # load data
    logger.info('Loading data...')
    arr_data, arr_labels = util.load_mnist('data')

    # save hyperparameters
    with open(os.path.join(args.model, 'params'), 'w') as f:
        for k, v in vars(args).items():
            print >> f, '{}\t{}'.format(k, v)

    # create batches
    logger.info('Creating batches...')
    batches = util.create_batches_mnist(arr_data, arr_labels, args.train_size, args.batch)

    label_dim = 10
    data_dim = 784
    active = getattr(chainer.functions, args.active)
    encoder = MnistEncoder(data_dim, label_dim, args.hidden, args.z, active)
    decoder = MnistDecoder(args.z, label_dim, args.hidden, data_dim, active)
    vae = Vae(encoder, decoder)

    vae.save_model_def(args.model)

    train.train_vae(vae, batches, optimizer, dest_dir=args.model, create_variables=_create_variables,
                    max_epoch=args.epoch, gpu=args.gpu, save_every=args.save_every, get_status=_get_status,
                    alpha_init=args.alpha_init, alpha_delta=args.alpha_delta)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train VAE for MNIST analogy generation')

    parser.add_argument('model', help='destination of model')

    # NN architecture
    parser.add_argument('--z', type=int, default=128, help='dimension of hidden variable')
    parser.add_argument('--hidden', nargs='+', type=int, default=[512, 512], help='size of hidden layers of recognition/generation models')
    parser.add_argument('--active', default='relu', help='activation function between hidden layers')

    # training options
    parser.add_argument('--train-size', type=int, default=50000, help='number of training samples')
    parser.add_argument('--batch', type=int, default=128, help='batch size')
    parser.add_argument('--epoch', type=int, default=10, help='number of epochs to train')
    parser.add_argument('--optim', nargs='+', default=['Adam'], help='optimization method supported by chainer (optional arguments can be omitted)')
    parser.add_argument('--gpu', type=int, default=None, help='GPU ID (default: use CPU)')
    parser.add_argument('--save-every', type=int, default=1, help='save model every this number of epochs')

    parser.add_argument('--alpha-init', type=float, default=1., help='initial value of weight of KL loss')
    parser.add_argument('--alpha-delta', type=float, default=0., help='delta value of weight of KL loss')

    main(parser.parse_args())
