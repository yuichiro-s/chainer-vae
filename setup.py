# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

setup(
    name='encdec-vae',
    version='0.0.1',
    description='Implementation of Encoder-Decoder Variational Autoencoder (VAE) by chainer',
    packages=find_packages(),
    requires=['chainer', 'sklearn', 'matplotlib'],
)
