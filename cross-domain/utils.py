# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import random
import numpy as np
import codecs
import os

def shuffle_two_list(a, b):
	"""
	shuffle a, b simultaneously
	"""
	c = list(zip(a, b))
	random.shuffle(c)
	a, b = zip(*c)
	return a, b

def padding(X,padding_word):
	max_len = 0
	for x in X:
		if len(x) > max_len:
			max_len = len(x)
	padded_X = np.ones((len(X), max_len), dtype=np.int32) * padding_word
	for i in range(len(X)):
		for j in range(len(X[i])):
			padded_X[i, j] = X[i][j]
	return padded_X

def padding2(X,padding_word):
	max_len = 0
	for x in X:
		if len(x) > max_len:
			max_len = len(x)
	padded_X = np.ones((len(X), max_len,8), dtype=np.int32) * padding_word
	for i in range(len(X)):
		for j in range(len(X[i])):
			padded_X[i, j]=X[i][j]
	return padded_X

def padding3(X,padding_word):
	max_len = 0
	for x in X:
		if len(x) > max_len:
			max_len = len(x)
	padded_X = np.ones((len(X), max_len,9), dtype=np.int32) * padding_word
	for i in range(len(X)):
		for j in range(len(X[i])):
			padded_X[i, j] = X[i][j]
	return padded_X


def data_iterator(X, Y, batch_size, padding_word,shuffle=True):
	if shuffle == True:
		X, Y = shuffle_two_list(X, Y)

	data_len = len(X)
	batch_len = data_len / batch_size

	for i in range(batch_len):
		batch_X = X[i*batch_size:(i+1)*batch_size]
		batch_Y = Y[i*batch_size:(i+1)*batch_size]
		padded_X = padding3(batch_X,padding_word)
		padded_Y = padding(batch_Y,3)
		yield padded_X,padded_Y

def data_iterator2(X, Y, batch_size, padding_word,shuffle=True):
	if shuffle == True:
		X, Y = shuffle_two_list(X, Y)

	X,dict_X=zip(*X)

	data_len = len(X)
	batch_len = data_len / batch_size

	for i in range(batch_len):
		batch_X = X[i*batch_size:(i+1)*batch_size]
		batch_dict=dict_X[i*batch_size:(i+1)*batch_size]
		batch_Y = Y[i*batch_size:(i+1)*batch_size]
		padded_X = padding3(batch_X,padding_word)
		padded_dict=padding2(batch_dict,padding_word)
		padded_Y = padding(batch_Y,3)
		yield padded_X, padded_dict,padded_Y

def true_pred(predict,length):
	true_pred=[]
	for i in range(len(predict)):
		true_pred.append(predict[i,:length[i]].tolist())
	return true_pred

def evaluate_word_PRF(y_pred, y):
	import itertools
	y_pred=list(itertools.chain.from_iterable(y_pred))
	y=list(itertools.chain.from_iterable(y))
	assert len(y_pred)==len(y)
	cor_num = 0
	yp_wordnum = y_pred.count(2) + y_pred.count(3)
	yt_wordnum = y.count(2) + y.count(3)
	start = 0
	for i in xrange(len(y)):
		if y[i] == 2 or y[i] == 3:
			flag = True
			for j in xrange(start, i + 1):
				if y[j] != y_pred[j]:
					flag = False
			if flag == True:
				cor_num += 1
			start = i + 1

	P = cor_num / float(yp_wordnum)
	R = cor_num / float(yt_wordnum)
	F = 2 * P * R / (P + R)
	return P, R, F

def convert_wordsegmentation(x,y,id2word,output_dir,output_file='result.txt'):
	if not os.path.exists('output'):
		os.mkdir('output')
	output_dir=os.path.join('output',output_dir)
	if not os.path.exists(output_dir):
		os.mkdir(output_dir)
	output_file=os.path.join(output_dir,output_file)
	f=codecs.open(output_file, 'w', encoding='utf-8')
	for i in range(len(x)):
		sentence=[]
		for j in range(len(x[i])):
			if y[i][j]==2 or y[i][j]==3:
				sentence.append(id2word[x[i][j][2]])
				sentence.append("  ")
			else:
				sentence.append(id2word[x[i][j][2]])
		f.write(''.join(sentence).strip()+'\n')
	f.close()























