# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import layers
from tensorflow.contrib import crf
from tensorflow.contrib import seq2seq

class BaselineModel(object):
    '''
    Baseline models
    BiLSTM+CRF and Stacked BiLSTM+CRF
    '''
    def __init__(self,vocab_size,word_dim,hidden_dim,
                 pad_word,init_embedding=None,
                 num_classes=4,clip=5,
                 lr=0.001,l2_reg_lamda=0.0001,num_layers=1,
                 rnn_cell='lstm',bi_direction=False
                 ):
        self.x=tf.placeholder(dtype=tf.int32,shape=[None,None,9],name='input_x')
        self.y=tf.placeholder(dtype=tf.int32,shape=[None,None],name='input_y')
        self.dropout_keep_prob=tf.placeholder(dtype=tf.float32,name='dropout_keep_prob')
        self.seq_length=tf.reduce_sum(tf.cast(tf.not_equal(self.x[:,:,2], tf.ones_like(self.x[:,:,2])*pad_word), tf.int32), 1)
        self.weights=tf.cast(tf.not_equal(self.x[:,:,2], tf.ones_like(self.x[:,:,2])*pad_word), tf.float32)
        self.batch_size = tf.shape(self.x)[0]

        if init_embedding is None:
            self.embedding=tf.get_variable(shape=[vocab_size,word_dim],dtype=tf.float32,name='embedding')
        else:
            self.embedding=tf.Variable(init_embedding,dtype=tf.float32,name='embedding')

        with tf.variable_scope('embedding'):
            x=tf.nn.embedding_lookup(self.embedding,self.x)
            x=tf.reshape(x,[self.batch_size,-1,9*word_dim])

        x=tf.nn.dropout(x,self.dropout_keep_prob)

        with tf.variable_scope('rnn_cell'):
            if rnn_cell=='lstm':
                print 'rnn_cell is lstm'
                self.fw_cell=rnn.BasicLSTMCell(hidden_dim)
                self.bw_cell=rnn.BasicLSTMCell(hidden_dim)
            else:
                print 'rnn_cell is gru'
                self.fw_cell=rnn.GRUCell(hidden_dim)
                self.bw_cell=rnn.GRUCell(hidden_dim)
            self.fw_cell = rnn.DropoutWrapper(self.fw_cell, output_keep_prob=self.dropout_keep_prob)
            self.bw_cell = rnn.DropoutWrapper(self.bw_cell, output_keep_prob=self.dropout_keep_prob)
            self.fw_multi_cell = rnn.MultiRNNCell([self.fw_cell] * num_layers)
            self.bw_multi_cell = rnn.MultiRNNCell([self.bw_cell] * num_layers)

        with tf.variable_scope('rnn'):
            if bi_direction:
                print 'bi_direction is true'
                (forward_output,backword_output),_=tf.nn.bidirectional_dynamic_rnn(
                    cell_fw=self.fw_multi_cell,
                    cell_bw=self.bw_multi_cell,
                    inputs=x,
                    sequence_length=self.seq_length,
                    dtype=tf.float32
                )
                output=tf.concat([forward_output,backword_output],axis=2)
            else:

                print 'bi_direction is false'
                forward_output,_=tf.nn.dynamic_rnn(
                    cell=self.fw_multi_cell,
                    inputs=x,
                    sequence_length=self.seq_length,
                    dtype=tf.float32
                )
                output=forward_output

        with tf.variable_scope('loss'):
            self.output=layers.fully_connected(
                inputs=output,
                num_outputs=num_classes,
                activation_fn=None
                )

            #crf
            log_likelihood, self.transition_params = crf.crf_log_likelihood(
                self.output, self.y, self.seq_length)
            loss = tf.reduce_mean(-log_likelihood)

        with tf.variable_scope('train_op'):
            self.optimizer=tf.train.AdamOptimizer(learning_rate=lr)
            tvars=tf.trainable_variables()
            l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tvars if v.get_shape().ndims > 1])
            self.loss=loss+l2_reg_lamda*l2_loss
            grads,_=tf.clip_by_global_norm(tf.gradients(self.loss,tvars),clip)
            self.train_op=self.optimizer.apply_gradients(zip(grads,tvars))

    def train_step(self,sess,x_batch,y_batch,dropout_keep_prob):
        feed_dict={
            self.x:x_batch,
            self.y:y_batch,
            self.dropout_keep_prob:dropout_keep_prob
        }
        _,loss=sess.run([self.train_op,self.loss],feed_dict)
        return loss

    def dev_step(self,sess,x_batch,y_batch):
        feed_dict={
            self.x:x_batch,
            self.y:y_batch,
            self.dropout_keep_prob:1.0
        }
        loss,lengths,unary_scores,transition_param=sess.run(
            [self.loss,self.seq_length,self.output, self.transition_params],feed_dict)
        predict=[]
        for unary_score,length in zip(unary_scores,lengths):
            viterbi_sequence, _=crf.viterbi_decode(unary_score[:length],transition_param)
            predict.append(viterbi_sequence)
        return loss,predict

    def predict_step(self,sess,x_batch):
        feed_dict={
            self.x:x_batch,
            self.dropout_keep_prob:1.0
        }
        lengths,unary_scores,transition_param = sess.run(
            [self.seq_length,self.output,self.transition_params], feed_dict)
        predict=[]
        for unary_score,length in zip(unary_scores,lengths):
            viterbi_sequence, _=crf.viterbi_decode(unary_score[:length],transition_param)
            predict.append(viterbi_sequence)
        return predict

if __name__ == '__main__':
    model=BaselineModel(64,16,8,0,num_layers=2,num_classes=6)
    data = [[1,2,0,0,0,0],
           [1,2,3,7,48,0],
           [1,2,6,1,1,0],
           [1,2,5,6,2,0],
           [1,2,3,3,0,0],
           [1,2,6,1,4,0]]
    y=[[1,2,3,4,5,4],
       [1,1,2,3,5,2],
       [3,0,2,3,4,0],
       [1,2,3,4,5,4],
       [1,1,2,3,5,2],
       [3,0,2,3,4,0]]

    data=np.ones(shape=[6,6,9])

    sess=tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    for i in xrange(100):
         _,l=sess.run([model.train_op,model.loss],feed_dict={model.x:data,model.y:y,model.dropout_keep_prob:1.0})
         print l
         print '----------------'
    print model.dev_step(sess,data,y)




