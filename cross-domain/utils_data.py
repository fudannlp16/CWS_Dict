# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from collections import Counter
import gensim
import numpy as np
import codecs
import os
import cPickle
import re
import preprocess
import random
random.seed(8)

UNK='U'
PAD='P'
START='S'
END='E'
TAGB,TAGI,TAGE,TAGS=0,1,2,3
DATA_PATH='data'

#convert word to tag
def word2tag(word):
    if len(word)==1:
        return [TAGS]
    if len(word)==2:
        return [TAGB,TAGE]
    tag=[]
    tag.append(TAGB)
    for i in range(1,len(word)-1):
        tag.append(TAGI)
    tag.append(TAGE)
    return tag

#get words from dictionaries
def get_words(general_words_path,domain_words_path=None):
    word_lists=dict()
    with codecs.open(general_words_path,'r','utf-8') as f:
        for line in f:
            line=line.strip().split()[0]
            word_lists[line]=1
    if domain_words_path is not None:
        with codecs.open(domain_words_path,'r','utf-8') as f:
            for line in f:
                line = line.strip().split()[0]
                word_lists[line] = 1
    return word_lists

#get dictionary from character and bigram to id
def get_word2id(filename,bigram_words=None,min_bw_frequence=0):
    filename=os.path.join(DATA_PATH,filename)
    x=[UNK,PAD,START,END]
    with codecs.open(filename,'r','utf-8') as f:
        for line in f:
            word_list=line.strip().split()
            for word in word_list:
                x.extend([c for c in word])
    bigrams=[]
    if bigram_words is not None:
        bigram_words=os.path.join('data',bigram_words)
        with codecs.open(bigram_words,'r','utf-8') as f:
            for line in f:
                com=line.strip().split()
                #filter some low frequency bigram words
                if int(com[1])>min_bw_frequence:
                    bigrams.append(com[0])
    x.extend(bigrams)
    counter = Counter(x)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))
    return word_to_id

#get reverse_dictionary from id to character or bigram
def build_reverse_dictionary(word_to_id):
    reverse_dictionary = dict(zip(word_to_id.values(), word_to_id.keys()))
    return reverse_dictionary

#get model's inputs
def get_train_data(filename=None,word2id=None,usebigram=True):
    filename=os.path.join(DATA_PATH,filename)
    x,y=[],[]
    with codecs.open(filename,'r','utf-8') as f:
        for line in f:
            word_list=line.strip().split()
            line_y=[]
            line_x=[]
            for word in word_list:
                line_y.extend(word2tag(word))
            y.append(line_y)
            line=re.sub(u'\s+','',line.strip())
            contexs=window(line)
            for contex in contexs:
                charx=[]
                #contex window
                charx.extend([word2id.get(c,word2id[UNK]) for c in contex])
                #bigram feature
                if usebigram:
                    charx.extend([word2id.get(bigram,word2id[UNK]) for bigram in preprocess.ngram(contex)])
                line_x.append(charx)
            x.append(line_x)
            assert len(line_x)==len(line_y)
    return x,y

def window(ustr,left=2,right=2):
    sent=''
    for i in range(left):
        sent+=START
    sent+=ustr
    for i in range(right):
        sent+=END
    windows=[]
    for i in range(len(ustr)):
        windows.append(sent[i:i+left+right+1])
    return windows

def tag_sentence(sentence,words):
    '''
    feature vector 
    '''
    word_list=words
    result=[]
    for i in range(len(sentence)):
        #fw
        word_tag=[]
        for j in range(4,0,-1):
            if (i-j)<0:
                word_tag.append(0)
                continue
            word=''.join(sentence[i-j:i+1])
            if word_list.get(word) is not None:
                word_tag.append(1)
            else:
                word_tag.append(0)
        #bw
        for j in range(1,5):
            if (i+j)>=len(sentence):
                word_tag.append(0)
                continue
            word=''.join(sentence[i:i+j+1])
            if word_list.get(word) is not None:
                word_tag.append(1)
            else:
                word_tag.append(0)
        result.append(word_tag)
    return result

def tag_documents(filename,words):
    filename=os.path.join(filename)
    result=[]
    with codecs.open(filename,'r','utf-8') as f:
        for line in f:
            word_list=line.strip().split()
            line_x=[]
            for word in word_list:
                line_x.extend([c for c in word])
            result.append(tag_sentence(line_x,words))
    return result

#get dictionary feature vector
def generate_dicttag(filename,general_words_path='general_dict',domain_words_path=None,p=1.0):
    filename=os.path.join(DATA_PATH,filename)
    general_words_path=os.path.join(DATA_PATH,general_words_path)
    if domain_words_path is None:
        domain_words_path=None
    else:
        domain_words_path=os.path.join(DATA_PATH,domain_words_path)
    word_lists=get_words(general_words_path,domain_words_path)
    all_words = word_lists.keys()
    random.shuffle(all_words)
    words=all_words[:int(len(word_lists) * p)]
    new_word_lists = dict()
    for word in words:
        new_word_lists[word] = 1
    data=tag_documents(filename,new_word_lists)
    return data

#get pre-trained embeddings
def get_embedding(word2id,size=100):
    fname='data/wordvec_'+str(size)
    init_embedding = np.zeros(shape=[len(word2id), size])
    pre_trained=gensim.models.KeyedVectors.load(fname)
    pre_trained_vocab = set([unicode(w.decode('utf8')) for w in pre_trained.vocab.keys()])
    c=0
    for word in word2id.keys():
        if len(word)==1:
            if word in pre_trained_vocab:
                init_embedding[word2id[word]]=pre_trained[word.encode('utf-8')]
            else:
                init_embedding[word2id[word]]=np.random.uniform(-0.5,0.5,size)
                c+=1
    for word in word2id.keys():
        if len(word)==2:
            init_embedding[word2id[word]]=(init_embedding[word2id[word[0]]]+init_embedding[word2id[word[1]]])/2
    init_embedding[word2id[PAD]]=np.zeros(shape=size)
    print 'oov character rate %f' % (float(c)/len(word2id))
    return init_embedding

if __name__ == '__main__':

    train_filename='train'
    test_filename='finance'
    word2id=get_word2id(train_filename,'train_bigram')
    id2word=dict([(y,x) for (x,y) in word2id.items()])
    print len(word2id)
    init_embedding=get_embedding(word2id,100)
    print init_embedding.shape
    x,y=get_train_data(test_filename,word2id)
    d=generate_dicttag(test_filename,domain_words_path=None)
    print x[0]
    print y[0]
    for i in range(len(x)):
        assert len(x[i])==len(y[i])
    print ''.join([id2word[x[2]] for x in x[0]])

    















