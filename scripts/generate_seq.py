from encdec_vae.nn.vae import Vae
from encdec_vae import vocabulary, util

import os

from chainer import cuda, Variable
import numpy as np


def main(args):
    if args.gpu is not None:
        cuda.get_device(args.gpu).use()

    # load model
    vae = Vae.load(args.model)
    label_dim = vae.encoder.rnn.feature_dim

    # load vocabulary
    model_base_path = os.path.dirname(args.model)
    vocab = vocabulary.Vocab.load(os.path.join(model_base_path, 'vocab'))

    while True:
        try:
            # process input
            in_str = raw_input('> ').decode('utf-8')
            es = in_str.split()
            label_id = int(es[0])
            ids = vocab.convert_ids(es[1:])

            # create input
            xs = map(lambda d: Variable(np.asarray([d], dtype=np.int32), volatile='on'), ids)
            label_in = _create_label_var([label_id], label_dim)
            label_out = _create_label_var(range(label_dim), label_dim)

            score_ids, debug_info = vae.generate(xs, label_in, label_out, no_var=args.no_var, sample=args.sample, temp=args.temp, max_len=args.max_len)
            mu = debug_info['mu']
            ln_var = debug_info['ln_var']
            z = debug_info['z']
            for ids, score in score_ids:
                print u'{}\t{}'.format(score, u' '.join(vocab.convert_words(ids))).encode('utf-8')

        except Exception as ex:
            print 'Usage: <label ID> <space-separated tokens>'


def _create_label_var(label_list, label_dim):
    arr = util.convert_one_hot(label_list, dim=label_dim)
    return Variable(arr, volatile='on')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Sample sequences interactively')

    parser.add_argument('model', help='model path')
    parser.add_argument('--gpu', type=int, default=None, help='GPU ID (default: use CPU)')

    parser.add_argument('--sample', action='store_true', default=False, help='sample instead of greedy search')
    parser.add_argument('--temp', type=float, default=1.0, help='temperature parameter for softmax (temp -> inf is equivalent to "max")')
    parser.add_argument('--no-var', action='store_true', default=False, help='sample most likely z (mean)')
    parser.add_argument('--max-len', type=int, default=50, help='maximum length to generate')

    main(parser.parse_args())
