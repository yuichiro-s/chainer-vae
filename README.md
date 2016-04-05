# chainer-vae

Variational Auto-encoder (VAE) implementation by chainer.

## Installation
```
python setup.py install
pip install chainer
```

## MNIST analogy generation
```
python scripts/train_mnist.py model
python scripts/generate_mnist_analogy.py model/epoch9
```

## (TODO) sequence-to-sequence learning with monolingual/bilingual data
```
python scripts/train_seq.py model test_data/mf_vocab --mono test_data/mf_train_mono --bi test_data/mf_train_bi --epoch 1000 --save-every 100
python scripts/generate_seq.py model/epoch999
```
