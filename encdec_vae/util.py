import numpy as np
from chainer import cuda, Variable
import chainer.optimizers as O
from collections import defaultdict


def get_xp(gpu):
    if gpu is None:
        return np
    else:
        return cuda.cupy


def id2var(w_id, batch_size=1, gpu=None):
    xp = get_xp(gpu)
    arr = xp.asarray([w_id], dtype=np.int32).repeat(batch_size)
    return Variable(arr, volatile='auto')


def list2optimizer(lst):
    """Create chainer optimizer object from list of strings, such as ['SGD', '0.01']"""
    optim_name = lst[0]
    optim_args = map(float, lst[1:])
    optimizer = getattr(O, optim_name)(*optim_args)
    return optimizer


def load_mnist(data_home='data'):
    from sklearn.datasets import fetch_mldata
    mnist = fetch_mldata('MNIST original', data_home=data_home)
    assert mnist.data.shape == (70000, 784)
    arr_data = (mnist.data / 255.).astype(np.float32)
    arr_labels = mnist.target.astype(np.int32)
    return arr_data, arr_labels


def load_monolingual(path):
    data = []
    with open(path) as f:
        for line in f:
            line = line.strip().decode('utf-8')
            label, words = _label_and_words(line)
            data.append((label, words))
    return data


def load_bilingual(path):
    data = []
    src = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                line = line.decode('utf-8')
                label, words = _label_and_words(line)
                if src is None:
                    src = label, words
                else:
                    trg = label, words
                    data.append((src, trg))
                    src = None
    return data


def _label_and_words(line):
    label_str, sen_str = line.split(u'\t')
    label = int(label_str)
    words = sen_str.split()
    return label, words


def get_deterministic_idxs(size):
    import fractions
    prime = 10007
    assert fractions.gcd(size, prime) == 1
    return np.arange(size) * prime % size


def create_batches_mnist(arr_data, arr_labels, train_size, batch_size):
    idxs = get_deterministic_idxs(len(arr_data))[:train_size]

    data = arr_data[idxs].astype(np.float32)
    labels = arr_labels[idxs].astype(np.int32)
    labels_one_hot = convert_one_hot(labels, dim=10)

    batches = []
    for i in range(0, train_size, batch_size):
        x_data = data[i:i+batch_size]
        label_x_data = labels_one_hot[i:i+batch_size]
        t_data = None
        label_t_data = None
        batch = (x_data, label_x_data, t_data, label_t_data)
        batches.append(batch)

    return batches


def create_batches_monolingual(data, batch_size, vocab, label_dim):
    batches = []
    buckets = defaultdict(list)
    for label, words in data:
        ids = vocab.convert_ids(words)
        ids.append(vocab.eos_id)
        buckets[len(ids)].append((label, ids))
    for samples in buckets.values():
        for i in range(0, len(samples), batch_size):
            label_lst, ids_lst = zip(*samples[i:i+batch_size])
            label_arr = convert_one_hot(list(label_lst), label_dim)
            ids_arr = np.asarray(ids_lst, dtype=np.int32)
            batch = (ids_arr, label_arr, None, None)
            batches.append(batch)
    return batches


def create_batches_bilingual(data, batch_size, vocab, label_dim, dst_pad_size=5):
    batches = []
    buckets = defaultdict(list)
    for (label_src, words_src), (label_trg, words_trg) in data:
        ids_src = vocab.convert_ids(words_src)
        ids_src.append(vocab.eos_id)
        ids_trg = vocab.convert_ids(words_trg)
        ids_trg.append(vocab.eos_id)
        while len(ids_trg) % dst_pad_size > 0:
            ids_trg.append(vocab.eos_id)
        buckets[len(ids_src), len(ids_trg)].append((label_src, ids_src, label_trg, ids_trg))
    for samples in buckets.values():
        for i in range(0, len(samples), batch_size):
            label_src_lst, ids_src_lst, label_trg_lst, ids_trg_lst = zip(*samples[i:i+batch_size])
            label_src_arr = convert_one_hot(list(label_src_lst), label_dim)
            ids_src_arr = np.asarray(ids_src_lst, dtype=np.int32)
            label_trg_arr = convert_one_hot(list(label_trg_lst), label_dim)
            ids_trg_arr = np.asarray(ids_trg_lst, dtype=np.int32)
            batch = (ids_src_arr, label_src_arr, ids_trg_arr, label_trg_arr)
            batches.append(batch)
    return batches


def convert_one_hot(labels, dim):
    return np.eye(dim).astype(np.float32)[labels]
