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
 python main.py --exp_name LM_Exp_Layer2 --exp_id 201 --num_tag 3 --model_name roberta-base --tgt_dm drugs  --src_dm drugs --batch_size 4  --epoch 5 --dropout 0  --cut_strategy 1 --n_layer 2  && \ 
 python main.py --exp_name LM_Exp_Layer2 --exp_id 205 --num_tag 3 --model_name roberta-base --ckpt LMs/ROBERTA/All/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 5 --dropout 0 --cut_strategy 1 --n_layer 2  && \ 
 python main.py --exp_name LM_Exp_Layer2 --exp_id 206 --num_tag 3 --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 5 --dropout 0 --cut_strategy 1 --n_layer 2  && \ 
 python main.py --exp_name LM_Exp_Layer2 --exp_id 210 --num_tag 3 --ckpt LMs/BERT/All/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 5 --dropout 0 --cut_strategy 1 --n_layer 2  && \ 


#Dropout 0.05
 python main.py --exp_name LM_Exp_Layer2 --exp_id 211 --num_tag 3 --model_name roberta-base --tgt_dm drugs  --src_dm drugs --batch_size 4  --epoch 5 --dropout 0.05  --cut_strategy 1 --n_layer 2  && \ 
 python main.py --exp_name LM_Exp_Layer2 --exp_id 215 --num_tag 3 --model_name roberta-base --ckpt LMs/ROBERTA/All/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 5 --dropout 0.05 --cut_strategy 1 --n_layer 2  && \ 
 python main.py --exp_name LM_Exp_Layer2 --exp_id 216 --num_tag 3 --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 5 --dropout 0.05 --cut_strategy 1 --n_layer 2  && \ 
 python main.py --exp_name LM_Exp_Layer2 --exp_id 220 --num_tag 3 --ckpt LMs/BERT/All/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 5 --dropout 0.05 --cut_strategy 1 --n_layer 2  && \ 

#Dropout 0.1
 python main.py --exp_name LM_Exp_Layer2 --exp_id 221 --num_tag 3 --model_name roberta-base --tgt_dm drugs  --src_dm drugs --batch_size 4  --epoch 5 --dropout 0.1  --cut_strategy 1 --n_layer 2  && \ 
 python main.py --exp_name LM_Exp_Layer2 --exp_id 225 --num_tag 3 --model_name roberta-base --ckpt LMs/ROBERTA/All/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 5 --dropout 0.1 --cut_strategy 1 --n_layer 2  && \ 
 python main.py --exp_name LM_Exp_Layer2 --exp_id 226 --num_tag 3 --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 5 --dropout 0.1 --cut_strategy 1 --n_layer 2  && \ 
 python main.py --exp_name LM_Exp_Layer2 --exp_id 230 --num_tag 3 --ckpt LMs/BERT/All/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 5 --dropout 0.1 --cut_strategy 1 --n_layer 2  && \ 


#Dropout 0.15
 python main.py --exp_name LM_Exp_Layer2 --exp_id 231 --num_tag 3 --model_name roberta-base --tgt_dm drugs  --src_dm drugs --batch_size 4  --epoch 5 --dropout 0.15  --cut_strategy 1 --n_layer 2  && \ 
 python main.py --exp_name LM_Exp_Layer2 --exp_id 235 --num_tag 3 --model_name roberta-base --ckpt LMs/ROBERTA/All/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 5 --dropout 0.15 --cut_strategy 1 --n_layer 2  && \ 
 python main.py --exp_name LM_Exp_Layer2 --exp_id 236 --num_tag 3 --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 5 --dropout 0.15 --cut_strategy 1 --n_layer 2  && \ 
 python main.py --exp_name LM_Exp_Layer2 --exp_id 240 --num_tag 3 --ckpt LMs/BERT/All/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 5 --dropout 0.15 --cut_strategy 1 --n_layer 2  && \ 

#Dropout 0.2
 python main.py --exp_name LM_Exp_Layer2 --exp_id 241 --num_tag 3 --model_name roberta-base --tgt_dm drugs  --src_dm drugs --batch_size 4  --epoch 5 --dropout 0.2  --cut_strategy 1 --n_layer 2  && \ 
 python main.py --exp_name LM_Exp_Layer2 --exp_id 245 --num_tag 3 --model_name roberta-base --ckpt LMs/ROBERTA/All/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 5 --dropout 0.2 --cut_strategy 1 --n_layer 2  && \ 
 python main.py --exp_name LM_Exp_Layer2 --exp_id 246 --num_tag 3 --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 5 --dropout 0.2 --cut_strategy 1 --n_layer 2  && \ 
 python main.py --exp_name LM_Exp_Layer2 --exp_id 250 --num_tag 3 --ckpt LMs/BERT/All/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 5 --dropout 0.2 --cut_strategy 1 --n_layer 2
