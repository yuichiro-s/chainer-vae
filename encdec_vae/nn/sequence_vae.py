from encdec_vae import util
from encdec_vae.nn import vae
from encdec_vae.nn.rnn import Rnn

import chainer.functions as F
import chainer.links as L
from chainer import ChainList, Variable
import numpy as np


class SequenceEncoder(vae.Encoder):

    def __init__(self, emb_dim, vocab_size, layer_dims, label_dim, z_dim):
        super(SequenceEncoder, self).__init__(
            rnn=Rnn(emb_dim, vocab_size, layer_dims, label_dim, suppress_output=True),
        )
        ls_mu = ChainList()
        ls_ln_var = ChainList()
        for d in layer_dims:
            ls_mu.add_link(L.Linear(d, z_dim))
            ls_ln_var.add_link(L.Linear(d, z_dim))
        self.add_link('ls_mu', ls_mu)
        self.add_link('ls_ln_var', ls_ln_var)

    def __call__(self, xs, label, gpu=None):
        batch_size = xs[0].data.shape[0]
        init_state = self.rnn.create_init_state(batch_size, gpu)
        final_state, _ = self.rnn(init_state, xs, label, gpu, prepend_eos=False)
        mu = 0
        ln_var = 0
        for l_mu, l_ln_var, (c, h) in zip(self.ls_mu, self.ls_ln_var, final_state):
            # currently, use only h
            mu += l_mu(h)
            ln_var += l_ln_var(h)
        return mu, ln_var


class SequenceDecoder(vae.Decoder):

    def __init__(self, emb_dim, vocab_size, layer_dims, label_dim, z_dim):
        super(SequenceDecoder, self).__init__(
            rnn=Rnn(emb_dim, vocab_size, layer_dims, label_dim, suppress_output=False),
        )
        ls_zh = ChainList()
        for d in layer_dims:
            ls_zh.add_link(L.Linear(z_dim, d))
        self.add_link('ls_zh', ls_zh)

    def __call__(self, z, label, ts, gpu=None):
        init_state = self._create_init_state_from_z(z, gpu)
        final_state, ys = self.rnn(init_state, ts, label, gpu, prepend_eos=True)
        rec_loss = 0
        for y, t in zip(ys, ts):
            rec_loss += F.softmax_cross_entropy(y, t)
        return rec_loss

    def generate(self, z, label, gpu=None, **kwargs):
        init_state = self._create_init_state_from_z(z, gpu)
        ids = self.rnn.generate(init_state, label, **kwargs)
        return ids

    def _create_init_state_from_z(self, z, gpu):
        xp = util.get_xp(gpu)
        batch_size = z.data.shape[0]
        init_state = []
        for l_zh in self.ls_zh:
            h = l_zh(z)
            d = h.data.shape[1]
            c_data = xp.zeros((batch_size, d), dtype=np.float32)
            c = Variable(c_data, volatile='auto')
            init_state.append((c, h))
        return init_state
