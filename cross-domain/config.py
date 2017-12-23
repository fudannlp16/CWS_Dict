# -*- coding: utf-8 -*-

class BaselineConfig:
    dropout_keep_prob = 0.8
    batch_size = 128
    word_dim=100
    hidden_dim=128
    num_classes = 4
    clip = 5
    lr = 0.01
    l2_reg_lamda = 0.0001
    num_layers = 1
    rnn_cell = 'lstm'
    bi_direction = True
    n_epoch=5

class DictConfig:
    dropout_keep_prob = 0.8
    batch_size = 128
    word_dim=100
    hidden_dim= 128
    hidden_dim2 = 160
    hyper_embed_size= 32
    num_classes = 4
    clip = 5
    lr = 0.01
    l2_reg_lamda = 0.0001
    num_layers = 1
    rnn_cell = 'lstm'
    bi_direction = True
    n_epoch=5







