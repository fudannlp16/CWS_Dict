# -*- coding: utf-8 -*-
from __future__ import division
import codecs

TRAIN='train.txt'
LITERATURE='literature.txt'
COMPUTER='computer.txt'
MEDICINE='medicine.txt'
FINANCE='finance.txt'

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

def count_oov(train_filename,test_filename):
    train_words=dict()
    with codecs.open(train_filename,'r','utf-8') as f:
        for line in f:
            words=line.strip().split()
            for word in words:
                train_words[word]=1
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

def statistic(train_filename,test_filename):
    sent=count_sent(test_filename)
    word=count_word(test_filename)
    char=count_char(test_filename)
    oov=count_oov(train_filename,test_filename)
    return sent,word,char,oov


if __name__ == '__main__':
    print 'Literature statistics'
    sent, word, char, oov=statistic(TRAIN,LITERATURE)
    print '#sent:%.1fK #word:%.1fK #char:%.1fK oov:%.1f%%' % (sent/1e3,word/1e3,char/1e3,oov)

    print 'Computer statistics'
    sent, word, char, oov=statistic(TRAIN,COMPUTER)
    print '#sent:%.1fK #word:%.1fK #char:%.1fK oov:%.1f%%' % (sent/1e3,word/1e3,char/1e3,oov)

    print 'Medicine statistics'
    sent, word, char, oov=statistic(TRAIN,MEDICINE)
    print '#sent:%.1fK #word:%.1fK #char:%.1fK oov:%.1f%%' % (sent/1e3,word/1e3,char/1e3,oov)

    print 'Finance statistics'
    sent, word, char, oov=statistic(TRAIN,FINANCE)
    print '#sent:%.1fK #word:%.1fK #char:%.1fK oov:%.1f%%' % (sent/1e3,word/1e3,char/1e3,oov)



