CUDA_VISIBLE_DEVICES=3 python train_dict.py --dataset pku --model_path pku2 --model DictHyperModel --is_train False
CUDA_VISIBLE_DEVICES=3 python train_dict.py --dataset msr --model_path msr2 --model DictHyperModel --is_train False --min_bg_freq 1
#CUDA_VISIBLE_DEVICES=3 python train_dict.py --dataset ctb --model_path ctb2 --model DictHyperModel --is_train False
CUDA_VISIBLE_DEVICES=3 python train_dict.py --dataset as --model_path as2 --model DictHyperModel --is_train False --min_bg_freq 1 --domain dict_2
CUDA_VISIBLE_DEVICES=3 python train_dict.py --dataset cityu --model_path cityu2 --model DictHyperModel --is_train False --domain dict_2