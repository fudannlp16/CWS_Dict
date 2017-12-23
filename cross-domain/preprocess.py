# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import codecs
import os
import re

TRAIN='original_data/train.txt'
LITERATURE='original_data/literature.txt'
COMPUTER='original_data/computer.txt'
MEDICINE='original_data/medicine.txt'
FINANCE='original_data/finance.txt'
OUTPUT_PATH='data'

rNUM = '(-|\+)?\d+((\.|·)\d+)?%?'
rENG = '[A-Za-z_.]+'
START='S'
END='E'

def strQ2B(ustring):
    """全角转半角"""
    rstring = ""
    for uchar in ustring:
        inside_code = ord(uchar)
        if inside_code == 12288:  # 全角空格直接转换
            inside_code = 32
        elif (inside_code >= 65281 and inside_code <= 65374):  # 全角字符（除空格）根据关系转化
            inside_code -= 65248
        rstring += unichr(inside_code)
    return rstring


def preprocess(input,output):
    output_filename = os.path.join(OUTPUT_PATH,output)
    sents=[]
    with codecs.open(input,'r','utf-8') as fin:
        with codecs.open(output_filename,'w','utf-8') as fout:
            for line in fin:
                sent=strQ2B(line).split( )
                new_sent=[]
                for word in sent:
                    word=re.sub(rNUM,'0',word)
                    word=re.sub(rENG,'X',word)
                    new_sent.append(word)
                sents.append(new_sent)
            for sent in sents:
                fout.write('  '.join(sent))
                fout.write('\n')

def ngram(ustr,n=2):
    ngram_list=[]
    for i in range(len(ustr)-n+1):
        ngram_list.append(ustr[i:i+n])
    return ngram_list

def bigram_words(dataset,window_size=2):
    dataset=os.path.join(OUTPUT_PATH,dataset)
    words=dict()
    start=''.join([START]*window_size)
    end=''.join([END]*window_size)
    with codecs.open(dataset,'r','utf-8') as f:
        for line in f:
            line=start+re.sub('\s+','',line.strip())+end
            for word in ngram(line,window_size):
                words[word]=words.get(word,0)+1
    with codecs.open(dataset+'_bigram','w','utf-8') as f:
        for k,v in words.items():
            f.write(k+' '+unicode(v)+'\n')

if __name__ == '__main__':
    if not os.path.exists(OUTPUT_PATH):
        os.mkdir(OUTPUT_PATH)
    #preprocess
    print 'start preprocess'
    preprocess(TRAIN,'train')
    preprocess(LITERATURE,'literature')
    preprocess(COMPUTER,'computer')
    preprocess(MEDICINE,'medicine')
    preprocess(FINANCE,'finance')
    #bigram
    print 'start bigram'
    bigram_words('train')







 
    