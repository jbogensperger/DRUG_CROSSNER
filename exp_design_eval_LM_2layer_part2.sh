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
 python main.py --exp_name LM_Exp_Layer2 --exp_id 251 --num_tag 3 --model_name roberta-base --tgt_dm drugs  --src_dm drugs --batch_size 4  --epoch 5 --dropout 0.25  --cut_strategy 1 --n_layer 2 && \ 
 python main.py --exp_name LM_Exp_Layer2 --exp_id 255 --num_tag 3 --model_name roberta-base --ckpt LMs/ROBERTA/All/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 5 --dropout 0.25 --cut_strategy 1 --n_layer 2 && \ 
 python main.py --exp_name LM_Exp_Layer2 --exp_id 256 --num_tag 3 --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 5 --dropout 0.25 --cut_strategy 1 --n_layer 2 && \ 
 python main.py --exp_name LM_Exp_Layer2 --exp_id 260 --num_tag 3 --ckpt LMs/BERT/All/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 5 --dropout 0.25 --cut_strategy 1 --n_layer 2 && \


#Dropout 0.3 && \ 
 python main.py --exp_name LM_Exp_Layer2 --exp_id 261 --num_tag 3 --model_name roberta-base --tgt_dm drugs  --src_dm drugs --batch_size 4  --epoch 5 --dropout 0.3  --cut_strategy 1 --n_layer 2 && \ 
 python main.py --exp_name LM_Exp_Layer2 --exp_id 265 --num_tag 3 --model_name roberta-base --ckpt LMs/ROBERTA/All/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 5 --dropout 0.3 --cut_strategy 1 --n_layer 2 && \ 
 python main.py --exp_name LM_Exp_Layer2 --exp_id 266 --num_tag 3 --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 5 --dropout 0.3 --cut_strategy 1 --n_layer 2 && \ 
 python main.py --exp_name LM_Exp_Layer2 --exp_id 270 --num_tag 3 --ckpt LMs/BERT/All/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 5 --dropout 0.3 --cut_strategy 1 --n_layer 2 && \

#Dropout 0.35 && \ 
 python main.py --exp_name LM_Exp_Layer2 --exp_id 271 --num_tag 3 --model_name roberta-base --tgt_dm drugs  --src_dm drugs --batch_size 4  --epoch 5 --dropout 0.35  --cut_strategy 1 --n_layer 2 && \ 
 python main.py --exp_name LM_Exp_Layer2 --exp_id 275 --num_tag 3 --model_name roberta-base --ckpt LMs/ROBERTA/All/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 5 --dropout 0.35 --cut_strategy 1 --n_layer 2 && \ 
 python main.py --exp_name LM_Exp_Layer2 --exp_id 276 --num_tag 3 --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 5 --dropout 0.35 --cut_strategy 1 --n_layer 2 && \ 
 python main.py --exp_name LM_Exp_Layer2 --exp_id 280 --num_tag 3 --ckpt LMs/BERT/All/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 5 --dropout 0.35 --cut_strategy 1 --n_layer 2 && \

#Dropout 0.4 && \ 
 python main.py --exp_name LM_Exp_Layer2 --exp_id 281 --num_tag 3 --model_name roberta-base --tgt_dm drugs  --src_dm drugs --batch_size 4  --epoch 5 --dropout 0.4  --cut_strategy 1 --n_layer 2 && \ 
 python main.py --exp_name LM_Exp_Layer2 --exp_id 285 --num_tag 3 --model_name roberta-base --ckpt LMs/ROBERTA/All/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 5 --dropout 0.4 --cut_strategy 1 --n_layer 2 && \ 
 python main.py --exp_name LM_Exp_Layer2 --exp_id 286 --num_tag 3 --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 5 --dropout 0.4 --cut_strategy 1 --n_layer 2 && \ 
 python main.py --exp_name LM_Exp_Layer2 --exp_id 290 --num_tag 3 --ckpt LMs/BERT/All/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 5 --dropout 0.4 --cut_strategy 1 --n_layer 2 && \

#Dropout 0.45 && \ 
 python main.py --exp_name LM_Exp_Layer2 --exp_id 291 --num_tag 3 --model_name roberta-base --tgt_dm drugs  --src_dm drugs --batch_size 4  --epoch 5 --dropout 0.45  --cut_strategy 1 --n_layer 2 && \ 
 python main.py --exp_name LM_Exp_Layer2 --exp_id 295 --num_tag 3 --model_name roberta-base --ckpt LMs/ROBERTA/All/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 5 --dropout 0.45 --cut_strategy 1 --n_layer 2 && \ 
 python main.py --exp_name LM_Exp_Layer2 --exp_id 296 --num_tag 3 --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 5 --dropout 0.45 --cut_strategy 1 --n_layer 2 && \ 
 python main.py --exp_name LM_Exp_Layer2 --exp_id 300 --num_tag 3 --ckpt LMs/BERT/All/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 5 --dropout 0.45 --cut_strategy 1 --n_layer 2 && \

#Dropout 0.5 && \ 
 python main.py --exp_name LM_Exp_Layer2 --exp_id 301 --num_tag 3 --model_name roberta-base --tgt_dm drugs  --src_dm drugs --batch_size 4  --epoch 5 --dropout 0.5  --cut_strategy 1 --n_layer 2 && \ 
 python main.py --exp_name LM_Exp_Layer2 --exp_id 305 --num_tag 3 --model_name roberta-base --ckpt LMs/ROBERTA/All/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 5 --dropout 0.5 --cut_strategy 1 --n_layer 2 && \ 
 python main.py --exp_name LM_Exp_Layer2 --exp_id 306 --num_tag 3 --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 5 --dropout 0.5 --cut_strategy 1 --n_layer 2 && \ 
 python main.py --exp_name LM_Exp_Layer2 --exp_id 310 --num_tag 3 --ckpt LMs/BERT/All/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 5 --dropout 0.5 --cut_strategy 1 --n_layer 2
 
 