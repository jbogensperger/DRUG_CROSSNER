# DRUG-CrossNER

This project is an adaptation of the original CrossNER implementation ( https://github.com/zliucr/CrossNER ). All further information about the original Project, paper citation etc. can be found in the next section.

Our DRUG-CrossNER project is focused on the detection of "drug" entities in Darknet Markets. Therefore we created our own Drug-NER dataset with over 3.500 item listings from the Dreammarket. 


## Project Execution
The experiments from the original work TODO{CITE original Master thesis} can be run with the following scripts.

please install a Conda envrionment with our "requirements.txt" by 

```console
conda create --name <env> --file requirements.txt
```

Activate the conda envirnoment and the Language models can be fine-tuned via the shell script "fine_tune_Language_Models.sh".

Afterwards the general BERT/RoBERTa Baselines for NER with a linear layer and dropout can be run via:

- "exp_design_eval_LM_part1.sh" and "exp_design_eval_LM_part2.sh" for the full training dataset.
- "exp_design_LM_fewShot_part1.sh" and "exp_design_LM_fewShot_part2.sh" for the FewShot scenario with using only 100 samples from the training dataset.

The Task adaption experiments can be run via:
- "exp_design_full_transfer.sh" - for the full trainin dataset.
- "exp_design_fewShot_transfer.sh" - for the FewShot scenario.


## Drug NER Dataset

The dataset for drug detection needs to be placed in "ner_data/drugs". Currently there are only place-holder files with a few examples present. For accessing the full dataset please contact TODO{NAME Contact}






## CrossNER
<img src="imgs/pytorch-logo-dark.png" width="10%"> [![](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<img align="right" src="imgs/HKUST.jpg" width="12%">

**NEW (2021/1/5):** Fixed several annotation errors (thanks for the help from Youliang Yuan).

**CrossNER: Evaluating Cross-Domain Named Entity Recognition** (**Accepted in AAAI-2021**) [[PDF]](https://arxiv.org/abs/2012.04373) 

CrossNER is a fully-labeled collected of named entity recognition (NER) data spanning over five diverse domains (Politics, Natural Science, Music, Literature, and Artificial Intelligence) with specialized entity categories for different domains. Additionally, CrossNER also includes unlabeled domain-related corpora for the corresponding five domains. We hope that our collected dataset (CrossNER) will catalyze research in the NER domain adaptation area.

You can have [**a quick overview of this paper**](https://zihanliu1026.medium.com/crossner-evaluating-cross-domain-named-entity-recognition-1a3ee2c1c42b) through our blog. If you use the dataset in an academic paper, please consider citing the following paper.
<pre>
@article{liu2020crossner,
      title={CrossNER: Evaluating Cross-Domain Named Entity Recognition}, 
      author={Zihan Liu and Yan Xu and Tiezheng Yu and Wenliang Dai and Ziwei Ji and Samuel Cahyawijaya and Andrea Madotto and Pascale Fung},
      year={2020},
      eprint={2012.04373},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
</pre>



### Dependency
- Install PyTorch (Tested in PyTorch 1.2.0 and Python 3.6)
- Install transformers (Tested in transformers 3.0.2)

### Domain-Adaptive Pre-Training (DAPT)

##### Configurations
- ```--train_data_file:``` The file path of the pre-training corpus.
- ```--output_dir:``` The output directory where the pre-trained model is saved.
- ```--model_name_or_path:``` Continue pre-training on which model.

```console
❱❱❱ python run_language_modeling.py --output_dir=politics_spanlevel_integrated --model_type=bert --model_name_or_path=bert-base-cased --do_train --train_data_file=corpus/politics_integrated.txt --mlm
```
This example is for span-level pre-training using integrated corpus in the politics domain. This code is modified based on run_language_modeling.py from [huggingface transformers](https://github.com/huggingface/transformers/tree/v3.1.0) (3.0.2).

### Baselines

##### Configurations
- ```--tgt_dm:``` Target domain that the model needs to adapt to.
- ```--conll:``` Using source domain data (News domain from CoNLL 2003) for pre-training.
- ```--joint:``` Jointly train using source and target domain data.
- ```--num_tag:``` Number of label types for the target domain (we put the details in src/dataloader.py).
- ```--ckpt:``` Checkpoint path to load the pre-trained model.
- ```--emb_file:``` Word-level embeddings file path.

#### Directly Fine-tune
Directly fine-tune the pre-trained model (span-level + integrated corpus) to the target domain (politics domain).
```console
❱❱❱ python main.py --exp_name politics_directly_finetune --exp_id 1 --num_tag 19 --ckpt politics_spanlevel_integrated/pytorch_model.bin --tgt_dm politics --batch_size 16
```


#### Pre-train then Fine-tune
Initialize the model with the pre-trained model (span-level + integrated corpus). Then fine-tune it to the target (politics) domain after pre-training on the source domain data.
```console
❱❱❱ python main.py --exp_name politics_pretrain_then_finetune --exp_id 1 --num_tag 19 --conll --ckpt politics_spanlevel_integrated/pytorch_model.bin --tgt_dm politics --batch_size 16
```

