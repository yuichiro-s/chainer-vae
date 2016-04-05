import sys

from encdec_vae import util
from encdec_vae.nn.vae import Vae

import os

from chainer import cuda, Variable
import numpy as np
import matplotlib.pyplot as plt


def main(args):
    if args.gpu is not None:
        cuda.get_device(args.gpu).use()

    # load data
    print >> sys.stderr, 'Loading data...'
    arr_data, arr_labels = util.load_mnist('data')

    # load model
    vae = Vae.load(args.model)

    # extract training data size
    train_size = None
    with open(os.path.join(os.path.dirname(args.model), 'params')) as f:
        for line in f.readlines():
            k, v = line.split()
            if k == 'train_size':
                train_size = int(v)
                break

    # exclude indices used in training data
    test_idxs = util.get_deterministic_idxs(len(arr_data))[train_size:]

    idxs = np.random.permutation(len(test_idxs))[:args.num]     # args.num samples not included in training data
    data = arr_data[test_idxs].astype(np.float32)[idxs]
    labels = arr_labels[test_idxs].astype(np.int32)[idxs]
    labels_one_hot = util.convert_one_hot(labels, dim=10)

    # labels to generate
    test_labels = np.eye(10).astype(np.float32)[np.expand_dims(np.arange(10), 0).repeat(args.num, axis=0).flatten()]

    xs = [Variable(data, volatile='on')]
    label_in = Variable(labels_one_hot, volatile='on')
    label_out = Variable(test_labels, volatile='on')

    print >> sys.stderr, 'Generating analogies...'
    x_rec, debug_info = vae.generate(xs, label_in, label_out)
    x_rec_reshape = np.reshape(x_rec.data, (args.num, 10, 28, 28))
    data_reshape = np.reshape(data, (args.num, 28, 28))

    print >> sys.stderr, 'Plotting...'
    plot(data_reshape, x_rec_reshape)


def plot(x, x_rec):
    width = 11
    height = x_rec.shape[0]
    fig, axis = plt.subplots(height, width, sharex=True, sharey=True)

    for i, (image_orig, image_recs) in enumerate(zip(x, x_rec)):
        # show original image
        ax = axis[i, 0]
        ax.imshow(image_orig, cmap=plt.cm.gray)
        ax.axis('off')

        # show generated images
        for j, image_rec in enumerate(image_recs):
            ax = axis[i, j + 1]
            ax.imshow(image_rec, cmap=plt.cm.gray)
            ax.axis('off')

    plt.show()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Generate MNIST analogies from trained model')

    parser.add_argument('model', help='model path')
    parser.add_argument('--num', type=int, default=3, help='number of samples to generate analogies')
    parser.add_argument('--gpu', type=int, default=None, help='GPU ID (default: use CPU)')

    main(parser.parse_args())
