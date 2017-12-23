# -*- coding: utf-8 -*-
from __future__ import division
import codecs

PKU_TRAIN='pku_training.utf8'
PKU_TEST='pku_test_gold.utf8'
MSR_TRAIN='msr_training.utf8'
MSR_TEST='msr_test_gold.utf8'
AS_TRAIN='as_training.utf8'
AS_TEST='as_test_gold.utf8'
CITYU_TRAIN='cityu_training.utf8'
CITYU_TEST='cityu_test_gold.utf8'
CTB_TRAIN='ctb_train'
CTB_DEV='ctb_dev'
CTB_TEST='ctb_test'

def count_sent(filename):
    count=0
    with codecs.open(filename,'r','utf-8') as f:
        for line in f:
            count+=1
    return count

def count_word(filename):
    count=0
    with codecs.open(filename,'r','utf-8') as f:
        for line in f:
            words=line.strip().split()
            count+=len(words)
    return count

def count_char(filename):
    count=0
    with codecs.open(filename,'r','utf-8') as f:
        for line in f:
            words=line.strip().split()
            for word in words:
                count+=len(word)
    return count

def count_oov(train_filename,test_filename,dev_filename=None):
    train_words=dict()
    with codecs.open(train_filename,'r','utf-8') as f:
        for line in f:
            words=line.strip().split()
            for word in words:
                train_words[word]=1
    if dev_filename is not None:
        with codecs.open(dev_filename,'r','utf-8') as f:
            for line in f:
                words = line.strip().split()
                for word in words:
                    train_words[word] = 1
    test_count=0
    test_common_count=0
    with codecs.open(test_filename,'r','utf-8') as f:
        for line in f:
            words=line.strip().split()
            for word in words:
                test_count+=1
                if train_words.get(word) is not None:
                    test_common_count+=1
    return (test_count-test_common_count)*100/test_count

def statistic(train_filename,test_filename,dev_filename=None):
    train_sent=count_sent(train_filename)
    train_word=count_word(train_filename)
    train_char=count_char(train_filename)
    if dev_filename is not None:
        train_sent+=count_sent(dev_filename)
        train_word+=count_word(dev_filename)
        train_char+=count_char(dev_filename)
    test_sent=count_sent(test_filename)
    test_word=count_word(test_filename)
    test_char=count_char(test_filename)
    oov=count_oov(train_filename,test_filename,dev_filename)
    return train_sent,train_word,train_char,test_sent,test_word,test_char,oov


if __name__ == '__main__':
    print 'PKU statistics'
    train_sent, train_word, train_char, test_sent, test_word, test_char, oov=statistic(PKU_TRAIN,PKU_TEST)
    print '#Train: sent:%.1fK #word:%.2fM #char:%.2fM' % (train_sent/1e3,train_word/1e6,train_char/1e6)
    print '#Test : sent:%.1fK #word:%.2fM #char:%.2fM' % (test_sent/1e3,test_word/1e6,test_char/1e6)
    print '#OOV: %.1f%%' % oov

    print 'MSR statistics'
    train_sent, train_word, train_char, test_sent, test_word, test_char, oov=statistic(MSR_TRAIN,MSR_TEST)
    print '#Train: sent:%.1fK #word:%.2fM #char:%.2fM' % (train_sent/1e3,train_word/1e6,train_char/1e6)
    print '#Test : sent:%.1fK #word:%.2fM #char:%.2fM' % (test_sent/1e3, test_word/1e6, test_char/1e6)
    print '#OOV: %.1f%%' % oov

    print 'AS statistics'
    train_sent, train_word, train_char, test_sent, test_word, test_char, oov=statistic(AS_TRAIN,AS_TEST)
    print '#Train: sent:%.1fK #word:%.2fM #char:%.2fM' % (train_sent/1e3,train_word/1e6,train_char/1e6)
    print '#Test : sent:%.1fK #word:%.2fM #char:%.2fM' % (test_sent/1e3, test_word/1e6, test_char/1e6)
    print '#OOV: %.1f%%' % oov

    print 'CITYU statistics'
    train_sent, train_word, train_char, test_sent, test_word, test_char, oov=statistic(CITYU_TRAIN,CITYU_TEST)
    print '#Train: sent:%.1fK #word:%.2fM #char:%.2fM' % (train_sent/1e3,train_word/1e6,train_char/1e6)
    print '#Test : sent:%.1fK #word:%.2fM #char:%.2fM' % (test_sent/1e3,test_word/1e6,test_char/1e6)
    print '#OOV: %.1f%%' % oov

    print 'CTB statistics'
    train_sent, train_word, train_char, test_sent, test_word, test_char, oov=statistic(CTB_TRAIN,CTB_TEST,CTB_DEV)
    print '#Train: sent:%.1fK #word:%.2fM #char:%.2fM' % (train_sent/1e3,train_word/1e6,train_char/1e6)
    print '#Test : sent:%.1fK #word:%.2fM #char:%.2fM' % (test_sent/1e3,test_word/1e6,test_char/1e6)
    print '#OOV: %.1f%%' % oov
