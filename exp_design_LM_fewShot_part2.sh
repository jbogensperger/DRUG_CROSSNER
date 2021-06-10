
#Second half of EXP V2

if [ -f "LMs/ROBERTA/Dreammarket/pytorch_model.bin" ] && [ -f "LMs/ROBERTA/Grams/pytorch_model.bin" ] && [ -f "LMs/ROBERTA/Wiki/pytorch_model.bin" ] && [ -f "LMs/ROBERTA/All/pytorch_model.bin" ]; then
    echo "$Roberta Language Models exist."
else 
    echo "Roberta Language model files are missing. please run fine_tune_Language_Models.sh"
	exit
fi
if [ -f "LMs/BERT/Dreammarket/pytorch_model.bin" ] && [ -f "LMs/BERT/Grams/pytorch_model.bin" ] && [ -f "LMs/BERT/Wiki/pytorch_model.bin" ] && [ -f "LMs/BERT/All/pytorch_model.bin" ] ; then
    echo "$BERT Language Models exist."
else 
    echo "BERT Language model files are missing. please run fine_tune_Language_Models.sh"
	exit
fi

#Dropout 0.25 && \ 
 python main.py --exp_name LM_Exp_few_V2 --exp_id 51 --num_tag 3 --model_name roberta-base --tgt_dm drugs  --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.25  --cut_strategy 1  --n_samples 100 && \ 
 python main.py --exp_name LM_Exp_few_V2 --exp_id 52 --num_tag 3 --model_name roberta-base --ckpt LMs/ROBERTA/Dreammarket/pytorch_model.bin --tgt_dm drugs  --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.25 --cut_strategy 1  --n_samples 100 && \ 
 python main.py --exp_name LM_Exp_few_V2 --exp_id 53 --num_tag 3 --model_name roberta-base --ckpt LMs/ROBERTA/Grams/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.25 --cut_strategy 1  --n_samples 100 && \ 
 python main.py --exp_name LM_Exp_few_V2 --exp_id 54 --num_tag 3 --model_name roberta-base --ckpt LMs/ROBERTA/Wiki/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.25 --cut_strategy 1  --n_samples 100 && \ 
 python main.py --exp_name LM_Exp_few_V2 --exp_id 55 --num_tag 3 --model_name roberta-base --ckpt LMs/ROBERTA/All/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.25 --cut_strategy 1  --n_samples 100 && \ 
 python main.py --exp_name LM_Exp_few_V2 --exp_id 56 --num_tag 3 --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.25 --cut_strategy 1  --n_samples 100 && \ 
 python main.py --exp_name LM_Exp_few_V2 --exp_id 57 --num_tag 3 --ckpt LMs/BERT/Dreammarket/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.25 --cut_strategy 1  --n_samples 100 && \ 
 python main.py --exp_name LM_Exp_few_V2 --exp_id 58 --num_tag 3 --ckpt LMs/BERT/Grams/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.25 --cut_strategy 1  --n_samples 100 && \ 
 python main.py --exp_name LM_Exp_few_V2 --exp_id 59 --num_tag 3 --ckpt LMs/BERT/Wiki/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.25 --cut_strategy 1  --n_samples 100 && \ 
 python main.py --exp_name LM_Exp_few_V2 --exp_id 60 --num_tag 3 --ckpt LMs/BERT/All/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.25 --cut_strategy 1  --n_samples 100 && \


#Dropout 0.3 && \ 
 python main.py --exp_name LM_Exp_few_V2 --exp_id 61 --num_tag 3 --model_name roberta-base --tgt_dm drugs  --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.3  --cut_strategy 1  --n_samples 100 && \ 
 python main.py --exp_name LM_Exp_few_V2 --exp_id 62 --num_tag 3 --model_name roberta-base --ckpt LMs/ROBERTA/Dreammarket/pytorch_model.bin --tgt_dm drugs  --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.3 --cut_strategy 1  --n_samples 100 && \ 
 python main.py --exp_name LM_Exp_few_V2 --exp_id 63 --num_tag 3 --model_name roberta-base --ckpt LMs/ROBERTA/Grams/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.3 --cut_strategy 1  --n_samples 100 && \ 
 python main.py --exp_name LM_Exp_few_V2 --exp_id 64 --num_tag 3 --model_name roberta-base --ckpt LMs/ROBERTA/Wiki/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.3 --cut_strategy 1  --n_samples 100 && \ 
 python main.py --exp_name LM_Exp_few_V2 --exp_id 65 --num_tag 3 --model_name roberta-base --ckpt LMs/ROBERTA/All/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.3 --cut_strategy 1  --n_samples 100 && \ 
 python main.py --exp_name LM_Exp_few_V2 --exp_id 66 --num_tag 3 --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.3 --cut_strategy 1  --n_samples 100 && \ 
 python main.py --exp_name LM_Exp_few_V2 --exp_id 67 --num_tag 3 --ckpt LMs/BERT/Dreammarket/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.3 --cut_strategy 1  --n_samples 100 && \ 
 python main.py --exp_name LM_Exp_few_V2 --exp_id 68 --num_tag 3 --ckpt LMs/BERT/Grams/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.3 --cut_strategy 1  --n_samples 100 && \ 
 python main.py --exp_name LM_Exp_few_V2 --exp_id 69 --num_tag 3 --ckpt LMs/BERT/Wiki/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.3 --cut_strategy 1  --n_samples 100 && \ 
 python main.py --exp_name LM_Exp_few_V2 --exp_id 70 --num_tag 3 --ckpt LMs/BERT/All/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.3 --cut_strategy 1  --n_samples 100 && \

#Dropout 0.35 && \ 
 python main.py --exp_name LM_Exp_few_V2 --exp_id 71 --num_tag 3 --model_name roberta-base --tgt_dm drugs  --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.35  --cut_strategy 1  --n_samples 100 && \ 
 python main.py --exp_name LM_Exp_few_V2 --exp_id 72 --num_tag 3 --model_name roberta-base --ckpt LMs/ROBERTA/Dreammarket/pytorch_model.bin --tgt_dm drugs  --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.35 --cut_strategy 1  --n_samples 100 && \ 
 python main.py --exp_name LM_Exp_few_V2 --exp_id 73 --num_tag 3 --model_name roberta-base --ckpt LMs/ROBERTA/Grams/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.35 --cut_strategy 1  --n_samples 100 && \ 
 python main.py --exp_name LM_Exp_few_V2 --exp_id 74 --num_tag 3 --model_name roberta-base --ckpt LMs/ROBERTA/Wiki/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.35 --cut_strategy 1  --n_samples 100 && \ 
 python main.py --exp_name LM_Exp_few_V2 --exp_id 75 --num_tag 3 --model_name roberta-base --ckpt LMs/ROBERTA/All/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.35 --cut_strategy 1  --n_samples 100 && \ 
 python main.py --exp_name LM_Exp_few_V2 --exp_id 76 --num_tag 3 --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.35 --cut_strategy 1  --n_samples 100 && \ 
 python main.py --exp_name LM_Exp_few_V2 --exp_id 77 --num_tag 3 --ckpt LMs/BERT/Dreammarket/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.35 --cut_strategy 1  --n_samples 100 && \ 
 python main.py --exp_name LM_Exp_few_V2 --exp_id 78 --num_tag 3 --ckpt LMs/BERT/Grams/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.35 --cut_strategy 1  --n_samples 100 && \ 
 python main.py --exp_name LM_Exp_few_V2 --exp_id 79 --num_tag 3 --ckpt LMs/BERT/Wiki/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.35 --cut_strategy 1  --n_samples 100 && \ 
 python main.py --exp_name LM_Exp_few_V2 --exp_id 80 --num_tag 3 --ckpt LMs/BERT/All/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.35 --cut_strategy 1  --n_samples 100 && \

#Dropout 0.4 && \ 
 python main.py --exp_name LM_Exp_few_V2 --exp_id 81 --num_tag 3 --model_name roberta-base --tgt_dm drugs  --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.4  --cut_strategy 1  --n_samples 100 && \ 
 python main.py --exp_name LM_Exp_few_V2 --exp_id 82 --num_tag 3 --model_name roberta-base --ckpt LMs/ROBERTA/Dreammarket/pytorch_model.bin --tgt_dm drugs  --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.4 --cut_strategy 1  --n_samples 100 && \ 
 python main.py --exp_name LM_Exp_few_V2 --exp_id 83 --num_tag 3 --model_name roberta-base --ckpt LMs/ROBERTA/Grams/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.4 --cut_strategy 1  --n_samples 100 && \ 
 python main.py --exp_name LM_Exp_few_V2 --exp_id 84 --num_tag 3 --model_name roberta-base --ckpt LMs/ROBERTA/Wiki/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.4 --cut_strategy 1  --n_samples 100 && \ 
 python main.py --exp_name LM_Exp_few_V2 --exp_id 85 --num_tag 3 --model_name roberta-base --ckpt LMs/ROBERTA/All/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.4 --cut_strategy 1  --n_samples 100 && \ 
 python main.py --exp_name LM_Exp_few_V2 --exp_id 86 --num_tag 3 --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.4 --cut_strategy 1  --n_samples 100 && \ 
 python main.py --exp_name LM_Exp_few_V2 --exp_id 87 --num_tag 3 --ckpt LMs/BERT/Dreammarket/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.4 --cut_strategy 1  --n_samples 100 && \ 
 python main.py --exp_name LM_Exp_few_V2 --exp_id 88 --num_tag 3 --ckpt LMs/BERT/Grams/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.4 --cut_strategy 1  --n_samples 100 && \ 
 python main.py --exp_name LM_Exp_few_V2 --exp_id 89 --num_tag 3 --ckpt LMs/BERT/Wiki/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.4 --cut_strategy 1  --n_samples 100 && \ 
 python main.py --exp_name LM_Exp_few_V2 --exp_id 90 --num_tag 3 --ckpt LMs/BERT/All/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.4 --cut_strategy 1  --n_samples 100 && \

#Dropout 0.45 && \ 
 python main.py --exp_name LM_Exp_few_V2 --exp_id 91 --num_tag 3 --model_name roberta-base --tgt_dm drugs  --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.45  --cut_strategy 1  --n_samples 100 && \ 
 python main.py --exp_name LM_Exp_few_V2 --exp_id 92 --num_tag 3 --model_name roberta-base --ckpt LMs/ROBERTA/Dreammarket/pytorch_model.bin --tgt_dm drugs  --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.45 --cut_strategy 1  --n_samples 100 && \ 
 python main.py --exp_name LM_Exp_few_V2 --exp_id 93 --num_tag 3 --model_name roberta-base --ckpt LMs/ROBERTA/Grams/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.45 --cut_strategy 1  --n_samples 100 && \ 
 python main.py --exp_name LM_Exp_few_V2 --exp_id 94 --num_tag 3 --model_name roberta-base --ckpt LMs/ROBERTA/Wiki/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.45 --cut_strategy 1  --n_samples 100 && \ 
 python main.py --exp_name LM_Exp_few_V2 --exp_id 95 --num_tag 3 --model_name roberta-base --ckpt LMs/ROBERTA/All/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.45 --cut_strategy 1  --n_samples 100 && \ 
 python main.py --exp_name LM_Exp_few_V2 --exp_id 96 --num_tag 3 --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.45 --cut_strategy 1  --n_samples 100 && \ 
 python main.py --exp_name LM_Exp_few_V2 --exp_id 97 --num_tag 3 --ckpt LMs/BERT/Dreammarket/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.45 --cut_strategy 1  --n_samples 100 && \ 
 python main.py --exp_name LM_Exp_few_V2 --exp_id 98 --num_tag 3 --ckpt LMs/BERT/Grams/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.45 --cut_strategy 1  --n_samples 100 && \ 
 python main.py --exp_name LM_Exp_few_V2 --exp_id 99 --num_tag 3 --ckpt LMs/BERT/Wiki/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.45 --cut_strategy 1  --n_samples 100 && \ 
 python main.py --exp_name LM_Exp_few_V2 --exp_id 100 --num_tag 3 --ckpt LMs/BERT/All/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.45 --cut_strategy 1  --n_samples 100 && \

#Dropout 0.5 && \ 
 python main.py --exp_name LM_Exp_few_V2 --exp_id 101 --num_tag 3 --model_name roberta-base --tgt_dm drugs  --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.5  --cut_strategy 1  --n_samples 100 && \ 
 python main.py --exp_name LM_Exp_few_V2 --exp_id 102 --num_tag 3 --model_name roberta-base --ckpt LMs/ROBERTA/Dreammarket/pytorch_model.bin --tgt_dm drugs  --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.5 --cut_strategy 1  --n_samples 100 && \ 
 python main.py --exp_name LM_Exp_few_V2 --exp_id 103 --num_tag 3 --model_name roberta-base --ckpt LMs/ROBERTA/Grams/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.5 --cut_strategy 1  --n_samples 100 && \ 
 python main.py --exp_name LM_Exp_few_V2 --exp_id 104 --num_tag 3 --model_name roberta-base --ckpt LMs/ROBERTA/Wiki/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.5 --cut_strategy 1  --n_samples 100 && \ 
 python main.py --exp_name LM_Exp_few_V2 --exp_id 105 --num_tag 3 --model_name roberta-base --ckpt LMs/ROBERTA/All/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.5 --cut_strategy 1  --n_samples 100 && \ 
 python main.py --exp_name LM_Exp_few_V2 --exp_id 106 --num_tag 3 --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.5 --cut_strategy 1  --n_samples 100 && \ 
 python main.py --exp_name LM_Exp_few_V2 --exp_id 107 --num_tag 3 --ckpt LMs/BERT/Dreammarket/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.5 --cut_strategy 1  --n_samples 100 && \ 
 python main.py --exp_name LM_Exp_few_V2 --exp_id 108 --num_tag 3 --ckpt LMs/BERT/Grams/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.5 --cut_strategy 1  --n_samples 100 && \ 
 python main.py --exp_name LM_Exp_few_V2 --exp_id 109 --num_tag 3 --ckpt LMs/BERT/Wiki/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.5 --cut_strategy 1  --n_samples 100 && \ 
 python main.py --exp_name LM_Exp_few_V2 --exp_id 110 --num_tag 3 --ckpt LMs/BERT/All/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.5 --cut_strategy 1  --n_samples 100 
