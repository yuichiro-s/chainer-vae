from encdec_vae import util

import chainer
import chainer.functions as F
from chainer import Chain, ChainList, Variable
import numpy as np


class Rnn(Chain):
    """Recurrent Neural Network"""

    def __init__(self, emb_dim, vocab_size, layer_dims, feature_dim, suppress_output, eos_id=0):
        """
        Recurrent Neural Network with multiple layers.
        in_dim -> layers[0] -> ... -> layers[-1] -> out_dim (optional)

        :param int emb_dim: dimension of embeddings
        :param int vocab_size: size of vocabulary
        :param layer_dims: dimensions of hidden layers
        :param int feature_dim: dimesion of external feature
        :type layer_dims: list of int
        :param bool suppress_output: whether to suppress output
        :param int eos_id: ID of <BOS> and <EOS>
        """
        super(Rnn, self).__init__(emb=F.EmbedID(vocab_size, emb_dim))

        self.emb_dim = emb_dim
        self.vocab_size = vocab_size
        self.layer_dims = layer_dims
        self.feature_dim = feature_dim
        self.suppress_output = suppress_output
        self.eos_id = eos_id

        # add hidden layer_dims
        ls_xh = ChainList()
        ls_hh = ChainList()
        ls_fh = ChainList()
        layer_dims = [emb_dim] + layer_dims
        for in_dim, out_dim in zip(layer_dims, layer_dims[1:]):
            ls_xh.add_link(F.Linear(in_dim, out_dim*4))
            ls_hh.add_link(F.Linear(out_dim, out_dim*4))
            ls_fh.add_link(F.Linear(feature_dim, out_dim*4))
        self.add_link('ls_xh', ls_xh)
        self.add_link('ls_hh', ls_hh)
        self.add_link('ls_fh', ls_fh)

        if not suppress_output:
            # add output layer
            self.add_link('l_y', F.Linear(layer_dims[-1], self.vocab_size))

    def step(self, state, x, feature):
        h = self.emb(x)
        new_state = []
        for l_xh, l_hh, l_fh, (last_c, last_h) in zip(self.ls_xh, self.ls_hh, self.ls_fh, state):
            h_in = l_xh(h) + l_hh(last_h) + l_fh(feature)
            c, h = F.lstm(last_c, h_in)
            new_state.append((c, h))
        y = None if self.suppress_output else self.l_y(h)
        return new_state, y

    def create_init_state(self, batch_size, gpu=None):
        """Creates initial state (hidden layers) filled with zeros."""
        state = []
        xp = util.get_xp(gpu)
        for layer_num, l in enumerate(self.layer_dims, 1):
            c_data = xp.zeros((batch_size, l), dtype=np.float32)
            h_data = xp.zeros((batch_size, l), dtype=np.float32)
            c = Variable(c_data, volatile='auto')
            h = Variable(h_data, volatile='auto')
            state.append((c, h))
        return state

    def __call__(self, state, xs, feature, gpu=None, prepend_eos=True):
        """Forward computation.

        :param state: initial state
        :type state: dict of (string, chainer.Variable)
        :param xs: list of input (EOS is prepended automatically)
        :type xs: list of chainer.Variable
        :return: final state (and unnormalized probabilities)
        """
        batch_size = xs[0].data.shape[0]
        x0 = util.id2var(self.eos_id, batch_size, gpu=gpu)
        ys = []
        if prepend_eos:
            xs = [x0] + xs
        for x in xs:
            state, y = self.step(state, x, feature)
            ys.append(y)
        return state, ys

    def generate(self, state, feature, sample, temp, max_len=50):
        """Generates sequence.

        :param state: initial state
        :type state: list of (chainer.Variable, chainer.Variable)
        :param chainer.Variable feature: external feature
        :param bool sample: sample instead of greedy search
        :param float temp: temparature parameter of softmax
        :param int max_len: maximum length of output
        :rtype: list of int
        """
        batch_size = feature.data.shape[0]

        # generation results
        score_list = [0.] * batch_size
        ids_list = []

        x = util.id2var(self.eos_id, batch_size)
        eos_samples = set()     # indices of samples where <EOS> has been generated

        for i in range(max_len):
            state, y = self.step(state, x, feature)

            probs = _softmax(y.data, temp)
            next_id_list = []
            for j in range(batch_size):
                if sample:
                    w_id = np.random.choice(probs.shape[1], p=probs[j])
                else:
                    w_id = np.argmax(probs[j])
                next_id_list.append(int(w_id))
                score_list[j] += np.log(probs[j, w_id])
                if w_id == self.eos_id:
                    eos_samples.add(j)
            ids_list.append(next_id_list)

            if len(eos_samples) == batch_size:
                # all finished
                break
            else:
                arr = np.asarray(next_id_list, dtype=np.int32)
                x = Variable(arr, volatile='on')

        results = []
        ids_list_t = zip(*ids_list)  # transpose
        for score, ids in zip(score_list, ids_list_t):
            # trim IDs after <EOS>
            if self.eos_id in ids:
                eos_idx = ids.index(self.eos_id)
                ids = ids[:eos_idx + 1]
            results.append((ids, score))
        assert len(results) == batch_size

        return results


def _softmax(d, temp):
    # d: batch x dim
    d *= temp
    d -= np.max(d, axis=1, keepdims=True)   # subtract max
    d_exp = np.exp(d)
    return d_exp / d_exp.sum(axis=1, keepdims=True)
