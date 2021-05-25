
### Language Model evaluation ###
python run_language_modeling.py --output_dir=LMs/ROBERTA/Dreammarket --model_type=roberta --model_name_or_path=roberta-base --do_train --train_data_file=corpus/DAPT_DreamMarket1.0.txt --mlm --per_device_train_batch_size=4 --per_device_eval_batch_size=4

## Only Grams Corpus 
python run_language_modeling.py --output_dir=LMs/ROBERTA/Grams --model_type=roberta --model_name_or_path=roberta-base --do_train --train_data_file=corpus/DAPT_Grams_1.0.txt --mlm --per_device_train_batch_size=4 --per_device_eval_batch_size=4

## Only Wikipedia Drug Corpus - TODO update wikipedia corpus?
python run_language_modeling.py --output_dir=LMs/ROBERTA/Wiki --model_type=roberta --model_name_or_path=roberta-base --do_train --train_data_file=corpus/Wiki_TextCorpusV1.0.txt --mlm --per_device_train_batch_size=4 --per_device_eval_batch_size=4

## All corpora merged - Grams, DreamMarket and Wikipedia 
python run_language_modeling.py --output_dir=LMs/ROBERTA/All --model_type=roberta --model_name_or_path=roberta-base --do_train --train_data_file=corpus/Merged_Corpora.txt --mlm --per_device_train_batch_size=4 --per_device_eval_batch_size=4



## Only DreamMarket DAPT Corpus (All rows which are not included in the dataset)
python run_language_modeling.py --output_dir=LMs/BERT/Dreammarket --model_type=bert --model_name_or_path=bert-base-cased --do_train --train_data_file=corpus/DAPT_DreamMarket1.0.txt --mlm --per_device_train_batch_size=4 --per_device_eval_batch_size=4

## Only Grams Corpus 
python run_language_modeling.py --output_dir=LMs/BERT/Grams --model_type=bert --model_name_or_path=bert-base-cased --do_train --train_data_file=corpus/DAPT_Grams_1.0.txt --mlm --per_device_train_batch_size=4 --per_device_eval_batch_size=4

## Only Wikipedia Drug Corpus - TODO update wikipedia corpus?
python run_language_modeling.py --output_dir=LMs/BERT/Wiki --model_type=bert --model_name_or_path=bert-base-cased --do_train --train_data_file=corpus/Wiki_TextCorpusV1.0.txt --mlm --per_device_train_batch_size=4 --per_device_eval_batch_size=4

## All corpora merged - Grams, DreamMarket and Wikipedia 
python run_language_modeling.py --output_dir=LMs/BERT/All --model_type=bert --model_name_or_path=bert-base-cased --do_train --train_data_file=corpus/Merged_Corpora.txt --mlm --per_device_train_batch_size=4 --per_device_eval_batch_size=4
