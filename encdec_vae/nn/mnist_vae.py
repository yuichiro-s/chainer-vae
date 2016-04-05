from encdec_vae.nn import vae

import chainer.functions as F
import chainer.links as L
from chainer import Chain, ChainList


class MnistEncoder(vae.Encoder):

    def __init__(self, in_dim, label_dim, hidden_dims, z_dim, active):
        super(MnistEncoder, self).__init__(
            l_mu=L.Linear(hidden_dims[-1], z_dim),
            l_ln_var=L.Linear(hidden_dims[-1], z_dim),
            mlp = _Mlp(in_dim+label_dim, hidden_dims, active),
        )

    def __call__(self, xs, label, gpu=None):
        x, = xs
        mlp_in = F.concat([x, label], axis=1)
        mlp_out = self.mlp(mlp_in)
        mu = self.l_mu(mlp_out)
        ln_var = self.l_ln_var(mlp_out)
        return mu, ln_var


class MnistDecoder(vae.Decoder):

    def __init__(self, z_dim, label_dim, hidden_dims, out_dim, active):
        super(MnistDecoder, self).__init__(
            l_last=L.Linear(hidden_dims[-1], out_dim),
            mlp = _Mlp(z_dim+label_dim, hidden_dims, active),
        )

    def __call__(self, z, label, ts, gpu=None):
        t, = ts
        batch_size = z.data.shape[0]
        mlp_in = F.concat([z, label], axis=1)
        mlp_out = self.mlp(mlp_in)
        x_rec = F.sigmoid(self.l_last(mlp_out))
        rec_loss = F.sum((t - x_rec) ** 2) / batch_size
        return rec_loss

    def generate(self, z, label, gpu=None, **kwargs):
        mlp_in = F.concat([z, label], axis=1)
        mlp_out = self.mlp(mlp_in)
        x_rec = F.sigmoid(self.l_last(mlp_out))
        return x_rec


class _Mlp(Chain):

    def __init__(self, in_dim, hidden_dims, active):
        super(_Mlp, self).__init__()
        self.active = active

        ds = [in_dim] + hidden_dims
        ls = ChainList()
        for d_in, d_out in zip(ds, ds[1:]):
            l = L.Linear(d_in, d_out)
            ls.add_link(l)
        self.add_link('ls', ls)

    def __call__(self, x):
        h = x
        for l in self.ls:
            h = self.active(l(h))
        return h
