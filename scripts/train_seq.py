#!/usr/bin/env python

from encdec_vae import util
from encdec_vae import vocabulary
from encdec_vae import train
from encdec_vae.nn.vae import Vae
from encdec_vae.nn.sequence_vae import SequenceEncoder, SequenceDecoder

import os
import logging

from chainer import cuda, Variable
import numpy as np


def _create_variables(x_data):
    vs = []
    for col in x_data.T:
        v = Variable(np.asarray(col))
        vs.append(v)
    return vs


def _get_status(batch):
    x_data, label_x_data, t_data, label_t_data = batch
    return [
        ('src', x_data.shape[0]),
        ('trg', t_data.shape[0] if t_data is not None else None),
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
    logger.info('Loading vocabulary...')
    vocab = vocabulary.Vocab.load(args.vocab, args.vocab_size)
    logger.info('Loading training data...')
    data_mono = util.load_monolingual(args.mono_data) if args.mono_data else None
    data_bi = util.load_bilingual(args.bi_data) if args.bi_data else None

    # save vocabulary
    vocab.save(os.path.join(args.model, 'vocab'))

    # count labels
    max_label = -1
    if data_mono:
        for label, _ in data_mono:
            max_label = max(max_label, label)
    if data_bi:
        for (label_src, _), (label_trg, _) in data_bi:
            max_label = max(max_label, label_src, label_trg)

    # save hyperparameters
    with open(os.path.join(args.model, 'params'), 'w') as f:
        for k, v in vars(args).items():
            print >> f, '{}\t{}'.format(k, v)

    # create batches
    logger.info('Creating batches...')
    label_dim = max_label + 1
    batches_mono = util.create_batches_monolingual(data_mono, args.batch, vocab, label_dim) if data_mono else []
    batches_bi = util.create_batches_bilingual(data_bi, args.batch, vocab, label_dim) if data_bi else []
    batches = batches_mono + batches_bi     # TODO: make mixing ratio adjustable

    # create VAE
    encoder = SequenceEncoder(args.emb, vocab.size(), args.hidden, label_dim, args.z)
    decoder = SequenceDecoder(args.emb, vocab.size(), args.hidden, label_dim, args.z)
    vae = Vae(encoder, decoder)
    vae.save_model_def(args.model)

    train.train_vae(vae, batches, optimizer, dest_dir=args.model, create_variables=_create_variables,
                    max_epoch=args.epoch, gpu=args.gpu, save_every=args.save_every, get_status=_get_status,
                    alpha_init=args.alpha_init, alpha_delta=args.alpha_delta)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train VAE for MNIST analogy generation')

    parser.add_argument('model', help='destination of model')
    parser.add_argument('vocab', help='path to vocabulary shared among languages')

    # data
    parser.add_argument('--mono-data', help='path to mono training data')
    parser.add_argument('--bi-data', help='path to bilingual training data')
    parser.add_argument('--vocab-size', type=int, default=-1, help='vocabulary size')

    # NN architecture
    parser.add_argument('--z', type=int, default=128, help='dimension of hidden variable')
    parser.add_argument('--emb', type=int, default=128, help='dimension of word embeddings')
    parser.add_argument('--hidden', nargs='+', type=int, default=[512, 512], help='size of hidden layers of recognition/generation models')

    # training options
    parser.add_argument('--batch', type=int, default=128, help='batch size')
    parser.add_argument('--epoch', type=int, default=10, help='number of epochs to train')
    parser.add_argument('--optim', nargs='+', default=['Adam'], help='optimization method supported by chainer (optional arguments can be omitted)')
    parser.add_argument('--gpu', type=int, default=None, help='GPU ID (default: use CPU)')
    parser.add_argument('--save-every', type=int, default=1, help='save model every this number of epochs')

    parser.add_argument('--alpha-init', type=float, default=1., help='initial value of weight of KL loss')
    parser.add_argument('--alpha-delta', type=float, default=0., help='delta value of weight of KL loss')

    main(parser.parse_args())
