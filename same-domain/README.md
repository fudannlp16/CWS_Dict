# Neural Networks Incorporating Dictionaries for Chinese Word Segmentation
Source codes and corpora for the Chinese word segmentation algorithm proposed in the following paper.

Qi Zhang, Xiaoyu Liu, Jinlan Fu. Neural Networks Incorporating Dictionaries for Chinese Word Segmentation. AAAI 2018 [PDF](http://jkx.fudan.edu.cn/~qzhang/paper/aaai2017-cws.pdf)

## Dependencies
* [Python 2.7](https://www.python.org/)
* [Tensorflow 1.0](https://www.tensorflow.org/)

## How to use
1.Firstly, preprocess the original datasets:

	python preprocess.py

The data directory would be:

    data
        dataset_train
        dataset_dev
        dataset_test
        dataset_train_bigram
        wordvec_100
        dict_1 (Simplified Chinese dictionary from jieba)
        dict_2 (Traditional Chinese dictionary from Taiwan version of jieba)
2.Then, set the hyperparameter of config.py according to the paper, and run:

    python train_dict.py --dataset  WHICH_DATASET --model  WHICH_MODEL --model_path  WHERE_SAVE_TRAINED_MODEL

dataset: pku,msr,as,cityu,ctb

model: DictConcatModel(Model-1) DictHyperModel(Model-II)

3.You can download our trained model from [here](https://drive.google.com/file/d/1vGQpSKamRRMQo-RWXViHq9173qDdNO2_/view?usp=sharing), and run the test script:

    ./scripts/test_dictmodel1.sh
    ./scripts/test_dictmodel2.sh

## Test Score
  | Model | PKU | MSR | CTB | AS |CITYU|
  |:------|:----|:----|:----|:---|:----|
  |Model-I| 96.2 | 97.6 | 96.1 |95.6|96.0|
  |Model-II|96.5 | 97.8	| 96.4 |95.9|96.3|





