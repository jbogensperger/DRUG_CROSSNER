# DRUG-CrossNER

This project is an adaptation of the original CrossNER implementation ( https://github.com/zliucr/CrossNER ). All further information about the original Project, paper citation etc. can be found in the next section.

Our DRUG-CrossNER project is focused on the detection of "drug" entities in Darknet Markets. Therefore we created our own Drug-NER dataset with over 3.500 item listings from the Dreammarket. 


## Project Execution
The experiments from the original work TODO{CITE original Master thesis} can be run with the following scripts.

please install a Conda envrionment with python=3.6 and activate it to install pytorch and transformers.

```console
conda create --name <env> python=3.6
```
And get Torch 1.7.1 via an install command from https://pytorch.org/get-started/previous-versions/ with the Cuda version you have on your server:

```console
#An example for CUDA 11.0
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
```
Alternative install commands for older Cuda Versions can be found (use only the command for YOUR cuda version).
```console
# CUDA 10.2
pip install torch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2

# CUDA 10.1
pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

# CUDA 9.2
pip install torch==1.7.1+cu92 torchvision==0.8.2+cu92 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

# CPU only
pip install torch==1.7.1+cpu torchvision==0.8.2+cpu torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

```

Finally please install transformers==3.5.1
```console
pip install transformers==3.5.1
```
## Test environment

one may want to test their set-up via the command
```console
python main.py --exp_name transfer --exp_id 1  --num_tag 3  --tgt_dm drugs  --src_dm drugs     --batch_size 4   --epoch 5
```

## Fine-Tune Language Models / Domain adaptation

BEFORE you can run all models, you need to fine-tune the language models! This will require the biggest part of the computing power. If you want to use exactly the models we used we provided the release with tag1.0 where a zip file with the language model folder with our binaries is included. just unpack the "LMs" folder directly in your project folder.

The Language models can be fine-tuned via the shell script "fine_tune_Language_Models.sh". This means we fine-tune BERT/RoBERTa to the target domain by using their specific learning tasks on text corpora from darknet markets and wikipedia articles about illicit drugs.

## Drug NER Dataset
Before starting with real experiments, you need to replace the sample entries for the drug domain in "ner_data/drugs" ("train.txt", "dev.txt", "test.txt") with the real full-size dataset. Currently there are only place-holder files with a few examples present. 

The dataset is currently only available via my github Address or an email to JBogen@gmx.at. You need to provide sufficient evidence to show your research interest for gaining access.

Once one has gained access to the dataset (3 files -train/dev/test.txt) it needs to be placed in "ner_data/drugs/". 




## Hyperparameter Tuning / creation of Baselines
After replacing the DRUG sample dataset with the real dataset, the general BERT/RoBERTa Baselines for Named Entity Recognition with a single linear layer and dropout can be run via:

- "exp_design_eval_LM_part1.sh" and "exp_design_eval_LM_part2.sh" for the full training dataset.
- "exp_design_LM_fewShot_part1.sh" and "exp_design_LM_fewShot_part2.sh" for the FewShot scenario with using only 100 samples from the training dataset.


## Running the best models

The best models from the experiments can be run via:

Best Model Few-Shot Scenario (BERT fine-tuned on ALL texts and dropout=0.5:
```console
python main.py --exp_name LM_Exp_few_V2 --exp_id 110 --num_tag 3 --ckpt LMs/BERT/All/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.5 --cut_strategy 1  --n_samples 100
```

Best Model Full Training Set Scenario (RoBERTa fine-tuned on ALL texts and dropout=0.5:
```console
python main.py --exp_name LM_Exp_V5_long --exp_id 225 --num_tag 3 --model_name roberta-base --ckpt LMs/ROBERTA/All/pytorch_model.bin --tgt_dm drugs --src_dm drugs --batch_size 4  --epoch 10 --dropout 0.1 --cut_strategy 1
```

## Task Adaptation

The Task adaption experiments can be run via the scripts found in the folder superseded_shell_scripts, since it was only relevant for the master thesis and not for the paper:
- "exp_design_full_transfer.sh" - for the full trainin dataset.
- "exp_design_fewShot_transfer.sh" - for the FewShot scenario.









## Information from the original CrossNER paper (condensed, the original info is longer)
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

