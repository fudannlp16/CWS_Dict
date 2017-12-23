CUDA_VISIBLE_DEVICES=3 python train_dict.py --dataset pku --model_path pku1 --model DictConcatModel --is_train False
CUDA_VISIBLE_DEVICES=3 python train_dict.py --dataset msr --model_path msr1 --model DictConcatModel --is_train False --min_bg_freq 1
#CUDA_VISIBLE_DEVICES=3 python train_dict.py --dataset ctb --model_path ctb1 --model DictConcatModel --is_train False
CUDA_VISIBLE_DEVICES=3 python train_dict.py --dataset as --model_path as1 --model DictConcatModel --is_train False --min_bg_freq 1 --domain dict_2
CUDA_VISIBLE_DEVICES=3 python train_dict.py --dataset cityu --model_path cityu1 --model DictConcatModel --is_train False --domain dict_2