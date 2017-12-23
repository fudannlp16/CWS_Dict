

# Neural Networks Incorporating Dictionaries for Chinese Word Segmentation
Source codes and corpora for the Chinese word segmentation algorithm proposed in the following paper.

Qi Zhang, Xiaoyu Liu, Jinlan Fu. Neural Networks Incorporating Dictionaries for Chinese Word Segmentation. AAAI 2018 [PDF](http://jkx.fudan.edu.cn/~qzhang/paper/aaai2017-cws.pdf)

## Dependencies
* [Python 2.7](https://www.python.org/)
* [Tensorflow 1.0](https://www.tensorflow.org/)

## How to use
1.Firstly, preprocess the original datasets. You should download SIGHAN2010 dataset and extract the archive to the original_data directory as:

    original_data:
        literature
        computer
        medicine
        finance
        train.txt
        statistic.py

You should run:

	python preprocess.py

The data directory would be:

    data
        literature
        literature_dict
        computer
        computer_dict
        medicine
        medicine_dict
        finance
        finance_dict
        wordvec_100
        general_dict
        source_dict
2.Then, set the hyperparameter of config.py according to the paper, and run:

    python train_dict.py --model  WHICH_MODEL --model_path  WHERE_SAVE_TRAINED_MODEL

model: DictConcatModel(Model-1) DictHyperModel(Model-II)

3.You can download our trained model from [here](), and run the test script:

    ./scripts/test_dictmodel1.sh
    ./scripts/test_dictmodel2.sh

## Test Score
  | Model | Literature | Computer | Medicine | Finance|
  |:------|:----|:----|:----|:---|
  |Model-I| 94.42 | 94.39 | 93.93 |95.70|
  |Model-I| 94.76 | 94.70.| 94.18 |96.06|


