### Test Language models Performance ###


#Dropout 0
CUDA_VISIBLE_DEVICES=2 python main.py --exp_name LM_Exp_few_V2 --exp_id 1 --num_tag 3 --model_name roberta-base --tgt_dm drugs  --src_dm drugs --batch_size 4  --epoch 10 --dropout 0  --cut_strategy 1  --n_samples 100
CUDA_VISIBLE_DEVICES=2 python main.py --exp_name LM_Exp_few_V2 --exp_id 2 --num_tag 3 --model_name roberta-base --ckpt LMs/ROBERTA/Dreammarket/pytorch_model.bin --tgt_dm drugs  --src_dm drugs --batch_size 4  --epoch 10 --dropout 0 --cut_strategy 1  --n_samples 100
CUDA_VISIBLE_DEVICES=2 python main.py --exp_name LM_Exp_few_V2 --exp_id 3 --num_tag 3 --model_name roberta-base --ckpt LMs/ROBERTA/Grams/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 10 --dropout 0 --cut_strategy 1  --n_samples 100
CUDA_VISIBLE_DEVICES=2 python main.py --exp_name LM_Exp_few_V2 --exp_id 4 --num_tag 3 --model_name roberta-base --ckpt LMs/ROBERTA/Wiki/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 10 --dropout 0 --cut_strategy 1  --n_samples 100
CUDA_VISIBLE_DEVICES=2 python main.py --exp_name LM_Exp_few_V2 --exp_id 5 --num_tag 3 --model_name roberta-base --ckpt LMs/ROBERTA/All/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 10 --dropout 0 --cut_strategy 1  --n_samples 100
CUDA_VISIBLE_DEVICES=2 python main.py --exp_name LM_Exp_few_V2 --exp_id 6 --num_tag 3 --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 10 --dropout 0 --cut_strategy 1  --n_samples 100
CUDA_VISIBLE_DEVICES=2 python main.py --exp_name LM_Exp_few_V2 --exp_id 7 --num_tag 3 --ckpt LMs/BERT/Dreammarket/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 10 --dropout 0 --cut_strategy 1  --n_samples 100
CUDA_VISIBLE_DEVICES=2 python main.py --exp_name LM_Exp_few_V2 --exp_id 8 --num_tag 3 --ckpt LMs/BERT/Grams/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 10 --dropout 0 --cut_strategy 1  --n_samples 100
CUDA_VISIBLE_DEVICES=2 python main.py --exp_name LM_Exp_few_V2 --exp_id 9 --num_tag 3 --ckpt LMs/BERT/Wiki/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 10 --dropout 0 --cut_strategy 1  --n_samples 100
CUDA_VISIBLE_DEVICES=2 python main.py --exp_name LM_Exp_few_V2 --exp_id 10 --num_tag 3 --ckpt LMs/BERT/All/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 10 --dropout 0 --cut_strategy 1  --n_samples 100


#Dropout 0.05
CUDA_VISIBLE_DEVICES=2 python main.py --exp_name LM_Exp_few_V2 --exp_id 11 --num_tag 3 --model_name roberta-base --tgt_dm drugs  --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.05  --cut_strategy 1  --n_samples 100
CUDA_VISIBLE_DEVICES=2 python main.py --exp_name LM_Exp_few_V2 --exp_id 12 --num_tag 3 --model_name roberta-base --ckpt LMs/ROBERTA/Dreammarket/pytorch_model.bin --tgt_dm drugs  --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.05 --cut_strategy 1  --n_samples 100
CUDA_VISIBLE_DEVICES=2 python main.py --exp_name LM_Exp_few_V2 --exp_id 13 --num_tag 3 --model_name roberta-base --ckpt LMs/ROBERTA/Grams/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.05 --cut_strategy 1  --n_samples 100
CUDA_VISIBLE_DEVICES=2 python main.py --exp_name LM_Exp_few_V2 --exp_id 14 --num_tag 3 --model_name roberta-base --ckpt LMs/ROBERTA/Wiki/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.05 --cut_strategy 1  --n_samples 100
CUDA_VISIBLE_DEVICES=2 python main.py --exp_name LM_Exp_few_V2 --exp_id 15 --num_tag 3 --model_name roberta-base --ckpt LMs/ROBERTA/All/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.05 --cut_strategy 1  --n_samples 100
CUDA_VISIBLE_DEVICES=2 python main.py --exp_name LM_Exp_few_V2 --exp_id 16 --num_tag 3 --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.05 --cut_strategy 1  --n_samples 100
CUDA_VISIBLE_DEVICES=2 python main.py --exp_name LM_Exp_few_V2 --exp_id 17 --num_tag 3 --ckpt LMs/BERT/Dreammarket/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.05 --cut_strategy 1  --n_samples 100
CUDA_VISIBLE_DEVICES=2 python main.py --exp_name LM_Exp_few_V2 --exp_id 18 --num_tag 3 --ckpt LMs/BERT/Grams/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.05 --cut_strategy 1  --n_samples 100
CUDA_VISIBLE_DEVICES=2 python main.py --exp_name LM_Exp_few_V2 --exp_id 19 --num_tag 3 --ckpt LMs/BERT/Wiki/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.05 --cut_strategy 1  --n_samples 100
CUDA_VISIBLE_DEVICES=2 python main.py --exp_name LM_Exp_few_V2 --exp_id 20 --num_tag 3 --ckpt LMs/BERT/All/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.05 --cut_strategy 1  --n_samples 100

#Dropout 0.1
CUDA_VISIBLE_DEVICES=2 python main.py --exp_name LM_Exp_few_V2 --exp_id 21 --num_tag 3 --model_name roberta-base --tgt_dm drugs  --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.1  --cut_strategy 1  --n_samples 100
CUDA_VISIBLE_DEVICES=2 python main.py --exp_name LM_Exp_few_V2 --exp_id 22 --num_tag 3 --model_name roberta-base --ckpt LMs/ROBERTA/Dreammarket/pytorch_model.bin --tgt_dm drugs  --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.1 --cut_strategy 1  --n_samples 100
CUDA_VISIBLE_DEVICES=2 python main.py --exp_name LM_Exp_few_V2 --exp_id 23 --num_tag 3 --model_name roberta-base --ckpt LMs/ROBERTA/Grams/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.1 --cut_strategy 1  --n_samples 100
CUDA_VISIBLE_DEVICES=2 python main.py --exp_name LM_Exp_few_V2 --exp_id 24 --num_tag 3 --model_name roberta-base --ckpt LMs/ROBERTA/Wiki/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.1 --cut_strategy 1  --n_samples 100
CUDA_VISIBLE_DEVICES=2 python main.py --exp_name LM_Exp_few_V2 --exp_id 25 --num_tag 3 --model_name roberta-base --ckpt LMs/ROBERTA/All/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.1 --cut_strategy 1  --n_samples 100
CUDA_VISIBLE_DEVICES=2 python main.py --exp_name LM_Exp_few_V2 --exp_id 26 --num_tag 3 --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.1 --cut_strategy 1  --n_samples 100
CUDA_VISIBLE_DEVICES=2 python main.py --exp_name LM_Exp_few_V2 --exp_id 27 --num_tag 3 --ckpt LMs/BERT/Dreammarket/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.1 --cut_strategy 1  --n_samples 100
CUDA_VISIBLE_DEVICES=2 python main.py --exp_name LM_Exp_few_V2 --exp_id 28 --num_tag 3 --ckpt LMs/BERT/Grams/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.1 --cut_strategy 1  --n_samples 100
CUDA_VISIBLE_DEVICES=2 python main.py --exp_name LM_Exp_few_V2 --exp_id 29 --num_tag 3 --ckpt LMs/BERT/Wiki/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.1 --cut_strategy 1  --n_samples 100
CUDA_VISIBLE_DEVICES=2 python main.py --exp_name LM_Exp_few_V2 --exp_id 30 --num_tag 3 --ckpt LMs/BERT/All/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.1 --cut_strategy 1  --n_samples 100


#Dropout 0.15
CUDA_VISIBLE_DEVICES=2 python main.py --exp_name LM_Exp_few_V2 --exp_id 31 --num_tag 3 --model_name roberta-base --tgt_dm drugs  --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.15  --cut_strategy 1  --n_samples 100
CUDA_VISIBLE_DEVICES=2 python main.py --exp_name LM_Exp_few_V2 --exp_id 32 --num_tag 3 --model_name roberta-base --ckpt LMs/ROBERTA/Dreammarket/pytorch_model.bin --tgt_dm drugs  --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.15 --cut_strategy 1  --n_samples 100
CUDA_VISIBLE_DEVICES=2 python main.py --exp_name LM_Exp_few_V2 --exp_id 33 --num_tag 3 --model_name roberta-base --ckpt LMs/ROBERTA/Grams/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.15 --cut_strategy 1  --n_samples 100
CUDA_VISIBLE_DEVICES=2 python main.py --exp_name LM_Exp_few_V2 --exp_id 34 --num_tag 3 --model_name roberta-base --ckpt LMs/ROBERTA/Wiki/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.15 --cut_strategy 1  --n_samples 100
CUDA_VISIBLE_DEVICES=2 python main.py --exp_name LM_Exp_few_V2 --exp_id 35 --num_tag 3 --model_name roberta-base --ckpt LMs/ROBERTA/All/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.15 --cut_strategy 1  --n_samples 100
CUDA_VISIBLE_DEVICES=2 python main.py --exp_name LM_Exp_few_V2 --exp_id 36 --num_tag 3 --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.15 --cut_strategy 1  --n_samples 100
CUDA_VISIBLE_DEVICES=2 python main.py --exp_name LM_Exp_few_V2 --exp_id 37 --num_tag 3 --ckpt LMs/BERT/Dreammarket/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.15 --cut_strategy 1  --n_samples 100
CUDA_VISIBLE_DEVICES=2 python main.py --exp_name LM_Exp_few_V2 --exp_id 38 --num_tag 3 --ckpt LMs/BERT/Grams/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.15 --cut_strategy 1  --n_samples 100
CUDA_VISIBLE_DEVICES=2 python main.py --exp_name LM_Exp_few_V2 --exp_id 39 --num_tag 3 --ckpt LMs/BERT/Wiki/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.15 --cut_strategy 1  --n_samples 100
CUDA_VISIBLE_DEVICES=2 python main.py --exp_name LM_Exp_few_V2 --exp_id 40 --num_tag 3 --ckpt LMs/BERT/All/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.15 --cut_strategy 1  --n_samples 100

#Dropout 0.2
CUDA_VISIBLE_DEVICES=2 python main.py --exp_name LM_Exp_few_V2 --exp_id 41 --num_tag 3 --model_name roberta-base --tgt_dm drugs  --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.2  --cut_strategy 1  --n_samples 100
CUDA_VISIBLE_DEVICES=2 python main.py --exp_name LM_Exp_few_V2 --exp_id 42 --num_tag 3 --model_name roberta-base --ckpt LMs/ROBERTA/Dreammarket/pytorch_model.bin --tgt_dm drugs  --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.2 --cut_strategy 1  --n_samples 100
CUDA_VISIBLE_DEVICES=2 python main.py --exp_name LM_Exp_few_V2 --exp_id 43 --num_tag 3 --model_name roberta-base --ckpt LMs/ROBERTA/Grams/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.2 --cut_strategy 1  --n_samples 100
CUDA_VISIBLE_DEVICES=2 python main.py --exp_name LM_Exp_few_V2 --exp_id 44 --num_tag 3 --model_name roberta-base --ckpt LMs/ROBERTA/Wiki/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.2 --cut_strategy 1  --n_samples 100
CUDA_VISIBLE_DEVICES=2 python main.py --exp_name LM_Exp_few_V2 --exp_id 45 --num_tag 3 --model_name roberta-base --ckpt LMs/ROBERTA/All/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.2 --cut_strategy 1  --n_samples 100
CUDA_VISIBLE_DEVICES=2 python main.py --exp_name LM_Exp_few_V2 --exp_id 46 --num_tag 3 --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.2 --cut_strategy 1  --n_samples 100
CUDA_VISIBLE_DEVICES=2 python main.py --exp_name LM_Exp_few_V2 --exp_id 47 --num_tag 3 --ckpt LMs/BERT/Dreammarket/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.2 --cut_strategy 1  --n_samples 100
CUDA_VISIBLE_DEVICES=2 python main.py --exp_name LM_Exp_few_V2 --exp_id 48 --num_tag 3 --ckpt LMs/BERT/Grams/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.2 --cut_strategy 1  --n_samples 100
CUDA_VISIBLE_DEVICES=2 python main.py --exp_name LM_Exp_few_V2 --exp_id 49 --num_tag 3 --ckpt LMs/BERT/Wiki/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.2 --cut_strategy 1  --n_samples 100
CUDA_VISIBLE_DEVICES=2 python main.py --exp_name LM_Exp_few_V2 --exp_id 50 --num_tag 3 --ckpt LMs/BERT/All/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.2 --cut_strategy 1  --n_samples 100
