# -*- coding: utf-8 -*-
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
import time
import itertools
import models
import tensorflow as tf
import numpy as np
from config import *
from utils_data import *
from utils import *
from sklearn.model_selection import train_test_split

tf.flags.DEFINE_string('dataset','pku',"Dataset for evaluation")
tf.flags.DEFINE_string("model_path", 'baseline', "The filename of model path")
tf.flags.DEFINE_float("memory",1.0,"Allowing GPU memory growth")
tf.flags.DEFINE_bool('is_train',True,"Train or predict")
tf.flags.DEFINE_integer('min_bg_freq',0,'The mininum bigram_words frequency')
FLAGS = tf.flags.FLAGS

train_data_path=FLAGS.dataset+'_train'
dev_data_path=FLAGS.dataset+'_dev'
test_data_path=FLAGS.dataset+'_test'
bigram_words_path=FLAGS.dataset+'_train_bigram'
config=BaselineConfig

if FLAGS.dataset == 'pku':
    config.hidden_dim = 64
if FLAGS.dataset == 'msr' or FLAGS.dataset=='as':
    FLAGS.min_bg_freq = 1
if FLAGS.dataset == 'as' or FLAGS.dataset=='cityu':
    FLAGS.domain = 'dict_2'

def train():
    word2id = get_word2id(train_data_path,bigram_words=bigram_words_path,min_bw_frequence=FLAGS.min_bg_freq)
    X_train,y_train=get_train_data(train_data_path,word2id)
    X_valid,y_valid=get_train_data(dev_data_path,word2id)
    x_test, y_test = get_train_data(test_data_path, word2id)
    init_embedding = get_embedding(word2id,size=config.word_dim)

    print 'train_data_path: %s' % train_data_path
    print 'valid_data_path: %s' % dev_data_path
    print 'test_data_path: %s' % test_data_path
    print 'bigram_words_path: %s' % bigram_words_path
    print 'model_path: %s' % FLAGS.model_path
    print 'min_bg_freq: %d'% FLAGS.min_bg_freq

    print 'len(train_data): %d' % len(X_train)
    print 'len(valid_data): %d' % len(X_valid)
    print 'len(test_data): %d' % len(x_test)
    print 'init_embedding shape: [%d,%d]' % (init_embedding.shape[0], init_embedding.shape[1])
    print 'Train started!'
    tfConfig = tf.ConfigProto()
    tfConfig.gpu_options.per_process_gpu_memory_fraction = FLAGS.memory
    with tf.Session(config=tfConfig) as sess:
        model=models.BaselineModel(vocab_size=len(word2id),word_dim=config.word_dim,hidden_dim=config.hidden_dim,
                    pad_word=word2id[PAD],init_embedding=init_embedding,num_classes=config.num_classes,clip=config.clip,
                    lr=config.lr,l2_reg_lamda=config.l2_reg_lamda,num_layers=config.num_layers,rnn_cell=config.rnn_cell,
                    bi_direction=config.bi_direction)

        if not os.path.exists('checkpoints'):
            os.mkdir('checkpoints')
        checkpoints_model=os.path.join('checkpoints',FLAGS.model_path)
        saver = tf.train.Saver(tf.all_variables())

        ckpt = tf.train.get_checkpoint_state(checkpoints_model)
        if ckpt and ckpt.model_checkpoint_path:
            print 'restore from original model!'
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            sess.run(tf.global_variables_initializer())

        best_f1,best_e=0,0
        for epoch in xrange(config.n_epoch):
            start_time=time.time()

            #train
            train_loss=[]
            for step,(X,Y) in enumerate(data_iterator(X_train,y_train,config.batch_size,padding_word=word2id[PAD],shuffle=True)):
                loss=model.train_step(sess,X,Y,config.dropout_keep_prob)
                print 'epoch:%d>>%2.2f%%' % (epoch,config.batch_size*step*100.0/len(X_train)),'completed in %.2f (sec) <<\r' % (time.time()-start_time),
                sys.stdout.flush()
                train_loss.append(loss)
            train_loss=np.mean(train_loss,dtype=float)
            print 'Train Epoch %d loss %f' % (epoch, train_loss)

            #valid
            valid_loss=[]
            valid_pred=[]
            for i in range(0, len(X_valid), config.batch_size):
                input_x = X_valid[slice(i, i + config.batch_size)]
                input_x = padding3(input_x, word2id[PAD])
                y = y_valid[slice(i, i + config.batch_size)]
                y = padding(y,3)
                loss, predict= model.dev_step(sess, input_x, y)
                valid_loss.append(loss)
                valid_pred+=predict
            valid_loss=np.mean(valid_loss,dtype=float)
            P,R,F=evaluate_word_PRF(valid_pred,y_valid)
            print 'Valid Epoch %d loss %f' % (epoch,valid_loss)
            print 'P:%f R:%f F:%f' % (P,R,F)
            print '--------------------------------'

            if F>best_f1:
                best_f1=F
                best_e=0
                saver.save(sess,checkpoints_model)
            else:
                best_e+=1

            test_pred = []
            for i in range(0, len(x_test), config.batch_size):
                input_x = x_test[slice(i, i + config.batch_size)]
                input_x = padding3(input_x, word2id[PAD])
                y = y_test[slice(i, i + config.batch_size)]
                y = padding(y, 3)
                predict= model.predict_step(sess, input_x)
                test_pred += predict
            P, R, F = evaluate_word_PRF(test_pred, y_test)
            print 'Test: P:%f R:%f F:%f Best_dev_F:%f' % (P, R, F,best_f1)

            if best_e>4:
                print 'Early stopping'
                break

        print 'best_f1 on validset is %f' % best_f1

def predict():
    if FLAGS.model_path==None:
        raise 'Model path is None!'
    if FLAGS.dataset=='pku':
        config.hidden_dim=64
    word2id = get_word2id(train_data_path, bigram_words=bigram_words_path,min_bw_frequence=FLAGS.min_bg_freq)
    id2word=build_reverse_dictionary(word2id)
    x_test, y_test = get_train_data(test_data_path, word2id)
    init_embedding = None

    print 'test_data_path: %s' % test_data_path
    print 'bigram_words_path: %s' % bigram_words_path
    print 'model_path: %s' % FLAGS.model_path
    print 'min_bg_freq: %d' % FLAGS.min_bg_freq

    with tf.Session() as sess:
        model=models.BaselineModel(vocab_size=len(word2id),word_dim=config.word_dim,hidden_dim=config.hidden_dim,
                    pad_word=word2id[PAD],init_embedding=init_embedding,num_classes=config.num_classes,clip=config.clip,
                    lr=config.lr,l2_reg_lamda=config.l2_reg_lamda,num_layers=config.num_layers,rnn_cell=config.rnn_cell,
                    bi_direction=config.bi_direction)
        checkpoints_model = os.path.join('checkpoints',FLAGS.model_path)
        saver = tf.train.Saver(tf.all_variables())
        ckpt = tf.train.get_checkpoint_state(checkpoints_model)
        if ckpt and ckpt.model_checkpoint_path:
            print 'test_start!'
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print '没有训练好的模型'
            exit()

        test_pred=[]
        for i in range(0,len(x_test),config.batch_size):
            input_x=x_test[slice(i,i+config.batch_size)]
            input_x=padding3(input_x,word2id[PAD])
            y = y_test[slice(i, i + config.batch_size)]
            y = padding(y, 3)
            predict= model.predict_step(sess, input_x)
            test_pred+=predict

        P,R,F=evaluate_word_PRF(test_pred,y_test)
        print '%s: P:%f R:%f F:%f' % (FLAGS.model_path,P,R,F)
        print '------------------------------------------'



if __name__ == '__main__':
    if FLAGS.is_train:
        train()
    else:
        predict()









