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

#Dropout 0.3 && \ 
 CUDA_VISIBLE_DEVICES=2 python main.py --exp_name LM_Exp_Layer2 --exp_id 265 --num_tag 3 --model_name roberta-base --ckpt LMs/ROBERTA/All/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 5 --dropout 0.3 --cut_strategy 1 --n_layer 2 && \ 
 CUDA_VISIBLE_DEVICES=2 python main.py --exp_name LM_Exp_Layer2 --exp_id 270 --num_tag 3 --ckpt LMs/BERT/All/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 5 --dropout 0.3 --cut_strategy 1 --n_layer 2 && \



#Dropout 0.4 && \ 
 CUDA_VISIBLE_DEVICES=2 python main.py --exp_name LM_Exp_Layer2 --exp_id 285 --num_tag 3 --model_name roberta-base --ckpt LMs/ROBERTA/All/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 5 --dropout 0.4 --cut_strategy 1 --n_layer 2 && \ 
 CUDA_VISIBLE_DEVICES=2 python main.py --exp_name LM_Exp_Layer2 --exp_id 290 --num_tag 3 --ckpt LMs/BERT/All/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 5 --dropout 0.4 --cut_strategy 1 --n_layer 2 && \


#Dropout 0.5 && \ 
 CUDA_VISIBLE_DEVICES=2 python main.py --exp_name LM_Exp_Layer2 --exp_id 305 --num_tag 3 --model_name roberta-base --ckpt LMs/ROBERTA/All/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 5 --dropout 0.5 --cut_strategy 1 --n_layer 2 && \ 
 CUDA_VISIBLE_DEVICES=2 python main.py --exp_name LM_Exp_Layer2 --exp_id 310 --num_tag 3 --ckpt LMs/BERT/All/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 5 --dropout 0.5 --cut_strategy 1 --n_layer 2



#Dropout 0.3 && \ 
 CUDA_VISIBLE_DEVICES=2 python main.py --exp_name LM_Exp_Layer2_long --exp_id 265 --num_tag 3 --model_name roberta-base --ckpt LMs/ROBERTA/All/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.3 --cut_strategy 1 --n_layer 2 && \ 
 CUDA_VISIBLE_DEVICES=2 python main.py --exp_name LM_Exp_Layer2_long --exp_id 270 --num_tag 3 --ckpt LMs/BERT/All/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.3 --cut_strategy 1 --n_layer 2 && \



#Dropout 0.4 && \ 
 CUDA_VISIBLE_DEVICES=2 python main.py --exp_name LM_Exp_Layer2_long --exp_id 285 --num_tag 3 --model_name roberta-base --ckpt LMs/ROBERTA/All/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.4 --cut_strategy 1 --n_layer 2 && \ 
 CUDA_VISIBLE_DEVICES=2 python main.py --exp_name LM_Exp_Layer2_long --exp_id 290 --num_tag 3 --ckpt LMs/BERT/All/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.4 --cut_strategy 1 --n_layer 2 && \


#Dropout 0.5 && \ 
 CUDA_VISIBLE_DEVICES=2 python main.py --exp_name LM_Exp_Layer2_long --exp_id 305 --num_tag 3 --model_name roberta-base --ckpt LMs/ROBERTA/All/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.5 --cut_strategy 1 --n_layer 2 && \ 
 CUDA_VISIBLE_DEVICES=2 python main.py --exp_name LM_Exp_Layer2_long --exp_id 310 --num_tag 3 --ckpt LMs/BERT/All/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.5 --cut_strategy 1 --n_layer 2
 
 