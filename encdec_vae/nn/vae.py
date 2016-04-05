import cPickle as pickle
import os
import copy

from chainer import Chain, Variable
import chainer.functions as F
from chainer.serializers import save_hdf5, load_hdf5


class Encoder(Chain):

    def __init__(self, **links):
        super(Encoder, self).__init__(**links)

    def __call__(self, xs, label, gpu):
        raise NotImplementedError


class Decoder(Chain):

    def __init__(self, **links):
        super(Decoder, self).__init__(**links)

    def __call__(self, z, label, ts, gpu):
        raise NotImplementedError

    def generate(self, z, label, gpu, **kwargs):
        raise NotImplementedError


class Vae(Chain):

    MODEL_DEF_NAME = 'model_def.pickle'

    def __init__(self, encoder, decoder):
        super(Vae, self).__init__(
            encoder=encoder,
            decoder=decoder,
        )

    def __call__(self, xs, label_in, ts, label_out):
        mu, ln_var = self.encoder(xs, label_in)
        z, kl = _infer_z(mu, ln_var)
        rec_loss = self.decoder(z, label_out, ts)
        return rec_loss, kl

    def generate(self, xs, label_in, label_out, no_var=False, **kwargs):
        mu, ln_var = self.encoder(xs, label_in)
        if no_var:
            # sample most likely z
            z = mu
        else:
            z = _infer_z(mu, ln_var)[0]

        # use same z for all samples
        label_out_num = label_out.data.shape[0]
        label_in_num = label_in.data.shape[0]
        assert label_out_num % label_in_num == 0
        rep = label_out_num / label_in_num
        z_data = z.data.repeat(rep, axis=0)
        z = Variable(z_data, volatile='on')

        debug_info = {
            'mu': mu,
            'ln_var': ln_var,
            'z': z,
        }
        return self.decoder.generate(z, label_out, **kwargs), debug_info

    def save_model_def(self, model_base_path):
        obj = copy.deepcopy(self)
        for p in obj.params():
            p.data = None
        path = os.path.join(model_base_path, self.MODEL_DEF_NAME)
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    def save(self, model_base_path, epoch):
        assert os.path.exists(os.path.join(model_base_path, self.MODEL_DEF_NAME)), 'Must call save_model_def() first'
        model_path = os.path.join(model_base_path, 'epoch{}'.format(epoch))
        save_hdf5(model_path, self)

    @classmethod
    def load(cls, path):
        model_base_path = os.path.dirname(path)
        model_def_path = os.path.join(model_base_path, cls.MODEL_DEF_NAME)
        with open(model_def_path, 'rb') as f:
            model = pickle.load(f)  # load model definition
            load_hdf5(path, model)  # load parameters
        return model


def _infer_z(mu, ln_var):
    batch_size = mu.data.shape[0]
    var = F.exp(ln_var)
    z = F.gaussian(mu, ln_var)
    kl = -F.sum(1 + ln_var - mu ** 2 - var) / 2
    kl /= batch_size
    return z, kl
