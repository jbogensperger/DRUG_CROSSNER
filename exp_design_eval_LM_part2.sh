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
 python main.py --exp_name LM_Exp_V5_long --exp_id 251 --num_tag 3 --model_name roberta-base --tgt_dm drugs  --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.25  --cut_strategy 1 && \ 
 python main.py --exp_name LM_Exp_V5_long --exp_id 252 --num_tag 3 --model_name roberta-base --ckpt LMs/ROBERTA/Dreammarket/pytorch_model.bin --tgt_dm drugs  --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.25 --cut_strategy 1 && \ 
 python main.py --exp_name LM_Exp_V5_long --exp_id 253 --num_tag 3 --model_name roberta-base --ckpt LMs/ROBERTA/Grams/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.25 --cut_strategy 1 && \ 
 python main.py --exp_name LM_Exp_V5_long --exp_id 254 --num_tag 3 --model_name roberta-base --ckpt LMs/ROBERTA/Wiki/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.25 --cut_strategy 1 && \ 
 python main.py --exp_name LM_Exp_V5_long --exp_id 255 --num_tag 3 --model_name roberta-base --ckpt LMs/ROBERTA/All/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.25 --cut_strategy 1 && \ 
 python main.py --exp_name LM_Exp_V5_long --exp_id 256 --num_tag 3 --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.25 --cut_strategy 1 && \ 
 python main.py --exp_name LM_Exp_V5_long --exp_id 257 --num_tag 3 --ckpt LMs/BERT/Dreammarket/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.25 --cut_strategy 1 && \ 
 python main.py --exp_name LM_Exp_V5_long --exp_id 258 --num_tag 3 --ckpt LMs/BERT/Grams/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.25 --cut_strategy 1 && \ 
 python main.py --exp_name LM_Exp_V5_long --exp_id 259 --num_tag 3 --ckpt LMs/BERT/Wiki/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.25 --cut_strategy 1 && \ 
 python main.py --exp_name LM_Exp_V5_long --exp_id 260 --num_tag 3 --ckpt LMs/BERT/All/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.25 --cut_strategy 1 && \


#Dropout 0.3 && \ 
 python main.py --exp_name LM_Exp_V5_long --exp_id 261 --num_tag 3 --model_name roberta-base --tgt_dm drugs  --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.3  --cut_strategy 1 && \ 
 python main.py --exp_name LM_Exp_V5_long --exp_id 262 --num_tag 3 --model_name roberta-base --ckpt LMs/ROBERTA/Dreammarket/pytorch_model.bin --tgt_dm drugs  --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.3 --cut_strategy 1 && \ 
 python main.py --exp_name LM_Exp_V5_long --exp_id 263 --num_tag 3 --model_name roberta-base --ckpt LMs/ROBERTA/Grams/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.3 --cut_strategy 1 && \ 
 python main.py --exp_name LM_Exp_V5_long --exp_id 264 --num_tag 3 --model_name roberta-base --ckpt LMs/ROBERTA/Wiki/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.3 --cut_strategy 1 && \ 
 python main.py --exp_name LM_Exp_V5_long --exp_id 265 --num_tag 3 --model_name roberta-base --ckpt LMs/ROBERTA/All/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.3 --cut_strategy 1 && \ 
 python main.py --exp_name LM_Exp_V5_long --exp_id 266 --num_tag 3 --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.3 --cut_strategy 1 && \ 
 python main.py --exp_name LM_Exp_V5_long --exp_id 267 --num_tag 3 --ckpt LMs/BERT/Dreammarket/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.3 --cut_strategy 1 && \ 
 python main.py --exp_name LM_Exp_V5_long --exp_id 268 --num_tag 3 --ckpt LMs/BERT/Grams/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.3 --cut_strategy 1 && \ 
 python main.py --exp_name LM_Exp_V5_long --exp_id 269 --num_tag 3 --ckpt LMs/BERT/Wiki/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.3 --cut_strategy 1 && \ 
 python main.py --exp_name LM_Exp_V5_long --exp_id 270 --num_tag 3 --ckpt LMs/BERT/All/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.3 --cut_strategy 1 && \

#Dropout 0.35 && \ 
 python main.py --exp_name LM_Exp_V5_long --exp_id 271 --num_tag 3 --model_name roberta-base --tgt_dm drugs  --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.35  --cut_strategy 1 && \ 
 python main.py --exp_name LM_Exp_V5_long --exp_id 272 --num_tag 3 --model_name roberta-base --ckpt LMs/ROBERTA/Dreammarket/pytorch_model.bin --tgt_dm drugs  --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.35 --cut_strategy 1 && \ 
 python main.py --exp_name LM_Exp_V5_long --exp_id 273 --num_tag 3 --model_name roberta-base --ckpt LMs/ROBERTA/Grams/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.35 --cut_strategy 1 && \ 
 python main.py --exp_name LM_Exp_V5_long --exp_id 274 --num_tag 3 --model_name roberta-base --ckpt LMs/ROBERTA/Wiki/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.35 --cut_strategy 1 && \ 
 python main.py --exp_name LM_Exp_V5_long --exp_id 275 --num_tag 3 --model_name roberta-base --ckpt LMs/ROBERTA/All/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.35 --cut_strategy 1 && \ 
 python main.py --exp_name LM_Exp_V5_long --exp_id 276 --num_tag 3 --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.35 --cut_strategy 1 && \ 
 python main.py --exp_name LM_Exp_V5_long --exp_id 277 --num_tag 3 --ckpt LMs/BERT/Dreammarket/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.35 --cut_strategy 1 && \ 
 python main.py --exp_name LM_Exp_V5_long --exp_id 278 --num_tag 3 --ckpt LMs/BERT/Grams/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.35 --cut_strategy 1 && \ 
 python main.py --exp_name LM_Exp_V5_long --exp_id 279 --num_tag 3 --ckpt LMs/BERT/Wiki/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.35 --cut_strategy 1 && \ 
 python main.py --exp_name LM_Exp_V5_long --exp_id 280 --num_tag 3 --ckpt LMs/BERT/All/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.35 --cut_strategy 1 && \

#Dropout 0.4 && \ 
 python main.py --exp_name LM_Exp_V5_long --exp_id 281 --num_tag 3 --model_name roberta-base --tgt_dm drugs  --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.4  --cut_strategy 1 && \ 
 python main.py --exp_name LM_Exp_V5_long --exp_id 282 --num_tag 3 --model_name roberta-base --ckpt LMs/ROBERTA/Dreammarket/pytorch_model.bin --tgt_dm drugs  --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.4 --cut_strategy 1 && \ 
 python main.py --exp_name LM_Exp_V5_long --exp_id 283 --num_tag 3 --model_name roberta-base --ckpt LMs/ROBERTA/Grams/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.4 --cut_strategy 1 && \ 
 python main.py --exp_name LM_Exp_V5_long --exp_id 284 --num_tag 3 --model_name roberta-base --ckpt LMs/ROBERTA/Wiki/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.4 --cut_strategy 1 && \ 
 python main.py --exp_name LM_Exp_V5_long --exp_id 285 --num_tag 3 --model_name roberta-base --ckpt LMs/ROBERTA/All/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.4 --cut_strategy 1 && \ 
 python main.py --exp_name LM_Exp_V5_long --exp_id 286 --num_tag 3 --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.4 --cut_strategy 1 && \ 
 python main.py --exp_name LM_Exp_V5_long --exp_id 287 --num_tag 3 --ckpt LMs/BERT/Dreammarket/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.4 --cut_strategy 1 && \ 
 python main.py --exp_name LM_Exp_V5_long --exp_id 288 --num_tag 3 --ckpt LMs/BERT/Grams/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.4 --cut_strategy 1 && \ 
 python main.py --exp_name LM_Exp_V5_long --exp_id 289 --num_tag 3 --ckpt LMs/BERT/Wiki/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.4 --cut_strategy 1 && \ 
 python main.py --exp_name LM_Exp_V5_long --exp_id 290 --num_tag 3 --ckpt LMs/BERT/All/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.4 --cut_strategy 1 && \

#Dropout 0.45 && \ 
 python main.py --exp_name LM_Exp_V5_long --exp_id 291 --num_tag 3 --model_name roberta-base --tgt_dm drugs  --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.45  --cut_strategy 1 && \ 
 python main.py --exp_name LM_Exp_V5_long --exp_id 292 --num_tag 3 --model_name roberta-base --ckpt LMs/ROBERTA/Dreammarket/pytorch_model.bin --tgt_dm drugs  --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.45 --cut_strategy 1 && \ 
 python main.py --exp_name LM_Exp_V5_long --exp_id 293 --num_tag 3 --model_name roberta-base --ckpt LMs/ROBERTA/Grams/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.45 --cut_strategy 1 && \ 
 python main.py --exp_name LM_Exp_V5_long --exp_id 294 --num_tag 3 --model_name roberta-base --ckpt LMs/ROBERTA/Wiki/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.45 --cut_strategy 1 && \ 
 python main.py --exp_name LM_Exp_V5_long --exp_id 295 --num_tag 3 --model_name roberta-base --ckpt LMs/ROBERTA/All/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.45 --cut_strategy 1 && \ 
 python main.py --exp_name LM_Exp_V5_long --exp_id 296 --num_tag 3 --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.45 --cut_strategy 1 && \ 
 python main.py --exp_name LM_Exp_V5_long --exp_id 297 --num_tag 3 --ckpt LMs/BERT/Dreammarket/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.45 --cut_strategy 1 && \ 
 python main.py --exp_name LM_Exp_V5_long --exp_id 298 --num_tag 3 --ckpt LMs/BERT/Grams/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.45 --cut_strategy 1 && \ 
 python main.py --exp_name LM_Exp_V5_long --exp_id 299 --num_tag 3 --ckpt LMs/BERT/Wiki/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.45 --cut_strategy 1 && \ 
 python main.py --exp_name LM_Exp_V5_long --exp_id 300 --num_tag 3 --ckpt LMs/BERT/All/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.45 --cut_strategy 1 && \

#Dropout 0.5 && \ 
 python main.py --exp_name LM_Exp_V5_long --exp_id 301 --num_tag 3 --model_name roberta-base --tgt_dm drugs  --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.5  --cut_strategy 1 && \ 
 python main.py --exp_name LM_Exp_V5_long --exp_id 302 --num_tag 3 --model_name roberta-base --ckpt LMs/ROBERTA/Dreammarket/pytorch_model.bin --tgt_dm drugs  --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.5 --cut_strategy 1 && \ 
 python main.py --exp_name LM_Exp_V5_long --exp_id 303 --num_tag 3 --model_name roberta-base --ckpt LMs/ROBERTA/Grams/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.5 --cut_strategy 1 && \ 
 python main.py --exp_name LM_Exp_V5_long --exp_id 304 --num_tag 3 --model_name roberta-base --ckpt LMs/ROBERTA/Wiki/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.5 --cut_strategy 1 && \ 
 python main.py --exp_name LM_Exp_V5_long --exp_id 305 --num_tag 3 --model_name roberta-base --ckpt LMs/ROBERTA/All/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.5 --cut_strategy 1 && \ 
 python main.py --exp_name LM_Exp_V5_long --exp_id 306 --num_tag 3 --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.5 --cut_strategy 1 && \ 
 python main.py --exp_name LM_Exp_V5_long --exp_id 307 --num_tag 3 --ckpt LMs/BERT/Dreammarket/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.5 --cut_strategy 1 && \ 
 python main.py --exp_name LM_Exp_V5_long --exp_id 308 --num_tag 3 --ckpt LMs/BERT/Grams/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.5 --cut_strategy 1 && \ 
 python main.py --exp_name LM_Exp_V5_long --exp_id 309 --num_tag 3 --ckpt LMs/BERT/Wiki/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.5 --cut_strategy 1 && \ 
 python main.py --exp_name LM_Exp_V5_long --exp_id 310 --num_tag 3 --ckpt LMs/BERT/All/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.5 --cut_strategy 1