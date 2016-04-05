import logging
import time
from collections import OrderedDict

from chainer import cuda, Variable


def train_vae(model, batches, optimizer, dest_dir, create_variables, max_epoch=None, gpu=None, save_every=1, get_status=None,
              alpha_init=1., alpha_delta=0.):
    """Common training procedure.

    :param model: model to train
    :param batches: training data
    :param optimizer: chainer optimizer
    :param dest_dir: destination directory
    :param create_variables: function to convert data to list of variable
    :param max_epoch: maximum number of epochs to train (None to train indefinitely)
    :param gpu: ID of GPU (None to use CPU)
    :param save_every: save every this number of epochs (first epoch and last epoch are always saved)
    :param get_status: function that takes batch and returns list of tuples of (name, value)
    :param alpha_init: initial value of alpha
    :param alpha_delta: change of alpha at every batch
    """
    if gpu is not None:
        # set up GPU
        cuda.get_device(gpu).use()
        model.to_gpu(gpu)

    logger = logging.getLogger()
    n_batches = len(batches)

    # set up optimizer
    optimizer.setup(model)

    # training loop
    epoch = 0
    alpha = alpha_init
    while True:
        if max_epoch is not None and epoch >= max_epoch:
            # terminate training
            break

        # train batches
        for i, batch in enumerate(batches):
            x_data, label_x_data, t_data, label_t_data = batch

            # copy data to GPU
            if gpu is not None:
                x_data = cuda.to_gpu(x_data)
                label_x_data = cuda.to_gpu(label_x_data)
                if t_data is not None:
                    t_data = cuda.to_gpu(t_data)
                    label_t_data = cuda.to_gpu(label_t_data)

            # create variable
            xs = create_variables(x_data)
            x_label = Variable(label_x_data)
            if t_data is None:
                # unsupervised
                ts = xs
                t_label = x_label
            else:
                # supervised
                ts = create_variables(t_data)
                t_label = Variable(label_t_data)

            # set new alpha
            alpha += alpha_delta
            alpha = min(alpha, 1.)
            alpha = max(alpha, 0.)

            time_start = time.time()

            optimizer.zero_grads()
            rec_loss, kl = model(xs, x_label, ts, t_label)
            loss = rec_loss + alpha * kl
            loss.backward()
            optimizer.update()

            time_end = time.time()
            time_delta = time_end - time_start

            # report training status
            status = OrderedDict()
            status['epoch'] = epoch
            status['batch'] = i
            status['prog'] = '{:.1%}'.format(float(i+1) / n_batches)
            status['time'] = int(time_delta * 1000)     # time in msec
            status['loss'] = '{:.4}'.format(float(loss.data))      # training loss
            status['alpha'] = alpha
            status['rec_loss'] = '{:.4}'.format(float(rec_loss.data))    # reconstruction loss
            status['kl'] = '{:.4}'.format(float(kl.data))    # KL-divergence loss
            if get_status is not None:
                status_lst = get_status(batch)
                for k, v in status_lst:
                    status[k] = v
            logger.info(_status_str(status))

        # save model
        if epoch % save_every == 0 or epoch == max_epoch:
            model.save(dest_dir, epoch)

        epoch += 1


def _status_str(status):
    lst = []
    for k, v in status.items():
        lst.append(k + ':')
        lst.append(str(v))
    return '\t'.join(lst)
