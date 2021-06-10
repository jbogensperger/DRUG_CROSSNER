### Test Language models Performance ###
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


#Dropout 0
 CUDA_VISIBLE_DEVICES=1 python main.py --exp_name LM_Exp_Layer2 --exp_id 205 --num_tag 3 --model_name roberta-base --ckpt LMs/ROBERTA/All/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 5 --dropout 0 --cut_strategy 1 --n_layer 2  && \ 
 CUDA_VISIBLE_DEVICES=1 python main.py --exp_name LM_Exp_Layer2 --exp_id 210 --num_tag 3 --ckpt LMs/BERT/All/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 5 --dropout 0 --cut_strategy 1 --n_layer 2  && \ 

#Dropout 0.1
 CUDA_VISIBLE_DEVICES=1 python main.py --exp_name LM_Exp_Layer2 --exp_id 225 --num_tag 3 --model_name roberta-base --ckpt LMs/ROBERTA/All/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 5 --dropout 0.1 --cut_strategy 1 --n_layer 2  && \ 
 CUDA_VISIBLE_DEVICES=1 python main.py --exp_name LM_Exp_Layer2 --exp_id 230 --num_tag 3 --ckpt LMs/BERT/All/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 5 --dropout 0.1 --cut_strategy 1 --n_layer 2  && \ 

#Dropout 0.2
 CUDA_VISIBLE_DEVICES=1 python main.py --exp_name LM_Exp_Layer2 --exp_id 245 --num_tag 3 --model_name roberta-base --ckpt LMs/ROBERTA/All/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 5 --dropout 0.2 --cut_strategy 1 --n_layer 2  && \ 
 CUDA_VISIBLE_DEVICES=1 python main.py --exp_name LM_Exp_Layer2 --exp_id 250 --num_tag 3 --ckpt LMs/BERT/All/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 5 --dropout 0.2 --cut_strategy 1 --n_layer 2


#Dropout 0
 CUDA_VISIBLE_DEVICES=1 python main.py --exp_name LM_Exp_Layer2_long --exp_id 205 --num_tag 3 --model_name roberta-base --ckpt LMs/ROBERTA/All/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 10 --dropout 0 --cut_strategy 1 --n_layer 2  && \ 
 CUDA_VISIBLE_DEVICES=1 python main.py --exp_name LM_Exp_Layer2_long --exp_id 210 --num_tag 3 --ckpt LMs/BERT/All/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 10 --dropout 0 --cut_strategy 1 --n_layer 2  && \ 

#Dropout 0.1
 CUDA_VISIBLE_DEVICES=1 python main.py --exp_name LM_Exp_Layer2_long --exp_id 225 --num_tag 3 --model_name roberta-base --ckpt LMs/ROBERTA/All/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.1 --cut_strategy 1 --n_layer 2  && \ 
 CUDA_VISIBLE_DEVICES=1 python main.py --exp_name LM_Exp_Layer2_long --exp_id 230 --num_tag 3 --ckpt LMs/BERT/All/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.1 --cut_strategy 1 --n_layer 2  && \ 

#Dropout 0.2
 CUDA_VISIBLE_DEVICES=1 python main.py --exp_name LM_Exp_Layer2_long --exp_id 245 --num_tag 3 --model_name roberta-base --ckpt LMs/ROBERTA/All/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.2 --cut_strategy 1 --n_layer 2  && \ 
 CUDA_VISIBLE_DEVICES=1 python main.py --exp_name LM_Exp_Layer2_long --exp_id 250 --num_tag 3 --ckpt LMs/BERT/All/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.2 --cut_strategy 1 --n_layer 2
