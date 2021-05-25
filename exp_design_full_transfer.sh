

### Source Target Domain evaluation ###
 
###  Directly Fine Tune  -     ###
CUDA_VISIBLE_DEVICES=0 python main.py --exp_name transfer --exp_id 21  --num_tag 3   --ckpt LMs/BERT/All/pytorch_model.bin                --tgt_dm drugs  --src_dm drugs     --batch_size 4   --epoch 5  --dropout 0.5
																					 
###  ConLL2004 - Drugs    -   ###                                                    
CUDA_VISIBLE_DEVICES=0 python main.py --exp_name transfer --exp_id 22  --num_tag 11  --ckpt LMs/BERT/All/pytorch_model.bin       --conll  --tgt_dm drugs  --src_dm conll     --batch_size 4   --epoch 5  --dropout 0.5
																					 
###  BTC - Drugs   - ###                                                             
CUDA_VISIBLE_DEVICES=0 python main.py --exp_name transfer --exp_id 23  --num_tag 9   --ckpt LMs/BERT/All/pytorch_model.bin       --btc    --tgt_dm drugs  --src_dm btc       --batch_size 4   --epoch 5  --dropout 0.5
																					 
###  WNUT - Drugs    -   ###                                                         
CUDA_VISIBLE_DEVICES=0 python main.py --exp_name transfer --exp_id 24  --num_tag 15  --ckpt LMs/BERT/All/pytorch_model.bin       --wnut   --tgt_dm drugs  --src_dm wnut      --batch_size 4   --epoch 5  --dropout 0.5
																					 
###  NUTOT - Drugs   -    ###                                                        
CUDA_VISIBLE_DEVICES=0 python main.py --exp_name transfer --exp_id 25  --num_tag 15  --ckpt LMs/BERT/All/pytorch_model.bin       --nutot  --tgt_dm drugs  --src_dm nutot     --batch_size 4   --epoch 5  --dropout 0.5
																					 
###  Distantly Supervised Drug-Wikipedia  - Drugs -   ###                            
CUDA_VISIBLE_DEVICES=0 python main.py --exp_name transfer --exp_id 26  --num_tag 3   --ckpt LMs/BERT/All/pytorch_model.bin       --wiki   --tgt_dm drugs  --src_dm wiki      --batch_size 4   --epoch 5  --dropout 0.5
