

### Source Target Domain evaluation ###
 
 ###  Directly Fine Tune  - NO DAPT/PRETRAINING AND No Dropout    ###
CUDA_VISIBLE_DEVICES=0 python main.py --exp_name final_few_transfer --exp_id 1  --num_tag 3        --tgt_dm drugs  --src_dm drugs     --batch_size 4     --epoch 10  --dropout 0 --n_samples 100

###  Directly Fine Tune  - NO DAPT/PRETRAINING    ###
CUDA_VISIBLE_DEVICES=0 python main.py --exp_name final_few_transfer --exp_id 102  --num_tag 3        --tgt_dm drugs  --src_dm drugs     --batch_size 4     --epoch 10  --dropout 0.5 --n_samples 100
 
###  Directly Fine Tune  -     ###
CUDA_VISIBLE_DEVICES=0 python main.py --exp_name final_few_transfer --exp_id 103  --num_tag 3   --ckpt LMs/BERT/All/pytorch_model.bin                 --tgt_dm drugs  --src_dm drugs     --batch_size 4     --epoch 10  --dropout 0.5 --n_samples 100
																																									  
###  ConLL2004 - Drugs    -   ###                                                                                                                                     
CUDA_VISIBLE_DEVICES=0 python main.py --exp_name final_few_transfer --exp_id 104  --num_tag 11  --ckpt LMs/BERT/All/pytorch_model.bin        --conll  --tgt_dm drugs  --src_dm conll     --batch_size 4     --epoch 10  --dropout 0.5 --n_samples 100
																																									  
###  BTC - Drugs   - ###                                                                                                                                              
CUDA_VISIBLE_DEVICES=0 python main.py --exp_name final_few_transfer --exp_id 105  --num_tag 9   --ckpt LMs/BERT/All/pytorch_model.bin        --btc    --tgt_dm drugs  --src_dm btc       --batch_size 4     --epoch 10  --dropout 0.5 --n_samples 100
																																									  
###  WNUT - Drugs    -   ###                                                                                                                                          
CUDA_VISIBLE_DEVICES=0 python main.py --exp_name final_few_transfer --exp_id 106  --num_tag 15  --ckpt LMs/BERT/All/pytorch_model.bin        --wnut   --tgt_dm drugs  --src_dm wnut      --batch_size 4     --epoch 10  --dropout 0.5 --n_samples 100
																																									  
###  NUTOT - Drugs   -    ###                                                                                                                                         
CUDA_VISIBLE_DEVICES=0 python main.py --exp_name final_few_transfer --exp_id 107  --num_tag 15  --ckpt LMs/BERT/All/pytorch_model.bin        --nutot  --tgt_dm drugs  --src_dm nutot     --batch_size 4     --epoch 10  --dropout 0.5 --n_samples 100
																																									  
###  Distantly Supervised Drug-Wikipedia  - Drugs -   ###                                                                                                             
CUDA_VISIBLE_DEVICES=0 python main.py --exp_name final_few_transfer --exp_id 108  --num_tag 3   --ckpt LMs/BERT/All/pytorch_model.bin        --wiki   --tgt_dm drugs  --src_dm wiki      --batch_size 4     --epoch 10  --dropout 0.5 --n_samples 100

###  Distantly Supervised Drug-Wikipedia  - Drugs -   ###                                                                                                             
CUDA_VISIBLE_DEVICES=0 python main.py --exp_name final_few_transfer --exp_id 109  --num_tag 3   --ckpt LMs/BERT/All/pytorch_model.bin        --wiki   --tgt_dm drugs  --src_dm wiki      --batch_size 4     --epoch 20  --dropout 0.5 --n_samples 100


#COMMENT: 100-109 were done on GPU 3 (TITAN) and 0-20 were done on GPU 2 or similiar GTX1080

