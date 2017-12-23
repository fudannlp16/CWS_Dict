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

tf.flags.DEFINE_float("memory",1.0,"Allowing GPU memory growth")
tf.flags.DEFINE_bool('is_train',True,"Train or predict")
tf.flags.DEFINE_string("model_path", 'baseline', "The filename of model path")
tf.flags.DEFINE_integer('min_bg_freq',0,'The mininum bigram_words frequency')
tf.flags.DEFINE_string('model','Baseline','')

#test
tf.flags.DEFINE_integer('epoch',1,"")

FLAGS = tf.flags.FLAGS

train_data_path='train'
bigram_words_path='train_bigram'
literature_path='literature'
computer_path='computer'
medicine_path='medicine'
finance_path='finance'

config=BaselineConfig

def train():
    word2id = get_word2id(train_data_path,bigram_words=bigram_words_path,min_bw_frequence=FLAGS.min_bg_freq)
    X_train,y_train=get_train_data(train_data_path,word2id)
    init_embedding = get_embedding(word2id,size=config.word_dim)
    # domain1
    X_test1, y_test1 = get_train_data(literature_path, word2id)
    # domain2
    X_test2, y_test2 = get_train_data(computer_path, word2id)
    # domain3
    X_test3, y_test3 = get_train_data(medicine_path, word2id)
    # domain4
    X_test4, y_test4 = get_train_data(finance_path, word2id)

    print 'train_data %s' % train_data_path
    print 'bigram %s' % bigram_words_path
    print 'model_path: %s' % FLAGS.model_path
    print 'min_bg_freq: %d'% FLAGS.min_bg_freq

    print 'len(train) %d ' % (len(X_train))
    print 'len(test) %d %d %d %d' % (len(X_test1),len(X_test2),len(X_test3),len(X_test4))
    print 'init_embedding shape [%d,%d]' % (init_embedding.shape[0], init_embedding.shape[1])
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

        best_f1,best_e1=0,0
        best_f2,best_e2=0,0
        best_f3,best_e3=0,0
        best_f4,best_e4=0,0
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
            saver.save(sess, checkpoints_model,epoch)

            def test(X_test,y_test,domain):
                test_pred = []
                for i in range(0, len(X_test), config.batch_size):
                    input_x = X_test[slice(i, i + config.batch_size)]
                    input_x = padding3(input_x, word2id[PAD])
                    y = y_test[slice(i, i + config.batch_size)]
                    y = padding(y, 3)

                    predict= model.predict_step(sess, input_x)
                    test_pred += predict
                P, R, F = evaluate_word_PRF(test_pred, y_test)
                print '%s Test: P:%f R:%f F:%f' % (domain, P, R, F)
                return F

            #domain1
            f1=test(X_test1,y_test1,'A')
            if best_f1<f1:
                best_f1=f1
                best_e1=epoch
            # domain2
            f2=test(X_test2,y_test2, 'B')
            if best_f2 < f2:
                best_f2=f2
                best_e2 = epoch
            #domain3
            f3=test(X_test3,y_test3,'C')
            if best_f3<f3:
                best_f3=f3
                best_e3=epoch
            #domain4
            f4=test(X_test4,y_test4,'D')
            if best_f4<f4:
                best_f4=f4
                best_e4=epoch
            print "best A:%f %d  best B: %f %d  best C %f %d  best D %f %d" %(best_f1,best_e1,best_f2,best_e2,best_f3,best_e3,best_f4,best_e4)
            print "******************************************************"

def predict():
    if FLAGS.model_path==None:
        raise 'Model path is None!'
    word2id = get_word2id(train_data_path, bigram_words=bigram_words_path,min_bw_frequence=FLAGS.min_bg_freq)
    id2word=build_reverse_dictionary(word2id)
    init_embedding = None

    # domain1
    X_test1, y_test1 = get_train_data(literature_path, word2id)
    # domain2
    X_test2, y_test2 = get_train_data(computer_path, word2id)
    # domain3
    X_test3, y_test3 = get_train_data(medicine_path, word2id)
    # domain4
    X_test4, y_test4 = get_train_data(finance_path, word2id)



    print 'len(test) %d  %d %d %d' % (len(X_test1),len(X_test2),len(X_test3),len(X_test4))
    print 'model %s' % FLAGS.model
    print 'bigram_words_path: %s' % bigram_words_path
    print 'model_path: %s' % FLAGS.model_path


    with tf.Session() as sess:
        model=models.BaselineModel(vocab_size=len(word2id),word_dim=config.word_dim,hidden_dim=config.hidden_dim,
                    pad_word=word2id[PAD],init_embedding=init_embedding,num_classes=config.num_classes,clip=config.clip,
                    lr=config.lr,l2_reg_lamda=config.l2_reg_lamda,num_layers=config.num_layers,rnn_cell=config.rnn_cell,
                    bi_direction=config.bi_direction)

        checkpoints_model = os.path.join('checkpoints',FLAGS.model_path)
        saver = tf.train.Saver(tf.all_variables())
        ckpt = tf.train.get_checkpoint_state(checkpoints_model)
        checkpoints_model = checkpoints_model+'-%d' % FLAGS.epoch

        print 'test_start!'
        saver.restore(sess, checkpoints_model)

        def test(X_test, y_test, domain):
            test_pred = []
            for i in range(0, len(X_test), config.batch_size):
                input_x = X_test[slice(i, i + config.batch_size)]
                input_x = padding3(input_x, word2id[PAD])
                y = y_test[slice(i, i + config.batch_size)]
                y = padding(y, 3)

                predict = model.predict_step(sess, input_x)
                test_pred += predict
            P, R, F = evaluate_word_PRF(test_pred, y_test)
            print '%s Test: P:%f R:%f F:%f' % (domain, P, R, F)
            return test_pred

        # domain1
        test_pred1 = test(X_test1, y_test1, 'Literature')
        convert_wordsegmentation(X_test1, test_pred1, id2word, FLAGS.model, literature_path)
        convert_wordsegmentation(X_test1, y_test1,id2word, FLAGS.model, literature_path+'_golden')
        # domain2
        test_pred2 = test(X_test2, y_test2, 'Computer  ')
        convert_wordsegmentation(X_test2, test_pred2,id2word,FLAGS.model, computer_path)
        convert_wordsegmentation(X_test2, y_test2, id2word, FLAGS.model, computer_path + '_golden')

        # domain3
        test_pred3 = test(X_test3, y_test3, 'Medicine  ')
        convert_wordsegmentation(X_test3, test_pred3, id2word, FLAGS.model, medicine_path)
        convert_wordsegmentation(X_test3, y_test3, id2word, FLAGS.model, medicine_path + '_golden')

        # domain4
        test_pred4 = test(X_test4, y_test4, 'Finance   ')
        convert_wordsegmentation(X_test4, test_pred4, id2word,FLAGS.model, finance_path)
        convert_wordsegmentation(X_test4, y_test4, id2word, FLAGS.model, finance_path + '_golden')


if __name__ == '__main__':
    if FLAGS.is_train:
        train()
    else:
        predict()









