# -*- coding: utf-8 -*-
from  __future__ import unicode_literals
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import cPickle
import tensorflow as tf
import models
import re
from config import DictConfig
import utils_data
import preprocess
import Queue

config=DictConfig
maps_file='checkpoints/maps_msr.pkl'
model_cache='checkpoints/msr2'
UNK='U'
PAD='P'
START='S'
END='E'
TAGB,TAGI,TAGE,TAGS=0,1,2,3

rNUM = '(-|\+)?\d+((\.|·)\d+)?%?'
rENG = '[A-Za-z_.]+'

class CWS_Dict:
    def __init__(self,model_name='DictHyperModel'):
        assert model_name in ['DictConcatModel','DictHyperModel']
        self.word2id,self.id2word,self.dict=cPickle.load(open(maps_file,'rb'))
        self.sess=tf.InteractiveSession()
        self.model = getattr(models,model_name)(vocab_size=len(self.word2id), word_dim=config.word_dim,
                                             hidden_dim=config.hidden_dim,
                                             pad_word=self.word2id[utils_data.PAD], init_embedding=None,
                                             num_classes=config.num_classes, clip=config.clip,
                                             lr=config.lr, l2_reg_lamda=config.l2_reg_lamda,
                                             num_layers=config.num_layers, rnn_cell=config.rnn_cell,
                                             bi_direction=config.bi_direction, hidden_dim2=config.hidden_dim2,
                                             hyper_embedding_size=config.hyper_embed_size)
        self.saver=tf.train.Saver(tf.all_variables())
        ckpt = tf.train.get_checkpoint_state(model_cache)
        self.saver.restore(self.sess,ckpt.model_checkpoint_path)


    def _findNum(self,sentence):
        results=Queue.Queue()
        for item in re.finditer(rNUM,sentence):
            results.put(item.group())
        return results

    def _findEng(self,sentence):
        results=Queue.Queue()
        for item in re.finditer(rENG,sentence):
            results.put(item.group())
        return results





    def _preprocess(self,sentence):
        original_sentence=[]
        new_sentence=[]
        num_lists = self._findNum(sentence)
        sentence2=re.sub(rNUM,'0',sentence)
        end_lists = self._findEng(sentence)
        sentence2=re.sub(rENG,'X',sentence2)
        original_sentence=[w for w in sentence2]
        new_sentence=preprocess.strQ2B(sentence2)
        return original_sentence,new_sentence,num_lists,end_lists

    def _input_from_line(self,sentence,user_words=None):
        line = sentence
        contexs = utils_data.window(line)
        line_x=[]
        for contex in contexs:
            charx = []
            # contex window
            charx.extend([self.word2id.get(c, self.word2id[utils_data.UNK]) for c in contex])
            # bigram feature
            charx.extend([self.word2id.get(bigram, self.word2id[utils_data.UNK]) for bigram in preprocess.ngram(contex)])
            line_x.append(charx)
        dict_feature=utils_data.tag_sentence(sentence,self.dict,user_words)
        return line_x,dict_feature


    def seg_sentence(self,sentence,user_words=None):
        original_sentence,new_sentence,num_lists,eng_lists=self._preprocess(sentence)
        line_x,dict_feature=self._input_from_line(new_sentence,user_words)
        predict=self.model.predict_step(self.sess,[line_x],[dict_feature])[0]
        seg_result=[]
        word=[]
        for char,tag in zip(original_sentence,predict):
            if char=='0':
                word.append(num_lists.get())
            elif char=='X':
                word.append(eng_lists.get())
            else:
                word.append(char)
            if tag==TAGE or tag==TAGS:
                seg_result.append(''.join(word))
                word=[]
        if len(word)>0:
            seg_result.append(''.join(word))
        return seg_result,predict

    def close_session(self):
        self.sess.close()

if __name__ == '__main__':
    model=CWS_Dict()
    while True:
        line = raw_input('\n请输入测试句子(0:Exit):\n'.encode('utf-8')).decode('utf-8')
        if line == '0':
            exit()
        words = raw_input('\n请输入自定义单词:\n'.encode('utf-8')).decode('utf-8').split()
        seg_result, tag=model.seg_sentence(line,words)
        print '切分结果:', ' '.join(seg_result)


















