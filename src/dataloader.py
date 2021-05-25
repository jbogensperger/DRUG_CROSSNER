import torch
import torch.nn as nn
import torch.utils.data as data
from torch.utils.data import DataLoader

import math
from tqdm import tqdm
import random
import logging

logger = logging.getLogger()

# from transformers import BertTokenizer
# bert_tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
from src.config import get_params

params = get_params()
from transformers import AutoTokenizer

auto_tokenizer = AutoTokenizer.from_pretrained(params.model_name)
pad_token_label_id = nn.CrossEntropyLoss().ignore_index


# domain2labels = {"politics": politics_labels, "science": science_labels, "music": music_labels, "literature": literature_labels, "ai": ai_labels }
# "drugs":conll_to_drugs, "single_drugs":single_drugs, "conll_to_drugs":conll_to_drugs, "wnut_to_drugs":wnut_to_drugs, "btc_to_drugs":btc_to_drugs

def remove_duplicates_with_order(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

def get_domain2labels(tgt_dm, src_dm):
    politics_labels = ['O', 'B-country', 'B-politician', 'I-politician', 'B-election', 'I-election', 'B-person',
                       'I-person', 'B-organisation', 'I-organisation', 'B-location', 'B-misc', 'I-location',
                       'I-country', 'I-misc', 'B-politicalparty', 'I-politicalparty', 'B-event', 'I-event']
    science_labels = ['O', 'B-scientist', 'I-scientist', 'B-person', 'I-person', 'B-university', 'I-university',
                      'B-organisation', 'I-organisation', 'B-country', 'I-country', 'B-location', 'I-location',
                      'B-discipline', 'I-discipline', 'B-enzyme', 'I-enzyme', 'B-protein', 'I-protein',
                      'B-chemicalelement', 'I-chemicalelement', 'B-chemicalcompound', 'I-chemicalcompound',
                      'B-astronomicalobject', 'I-astronomicalobject', 'B-academicjournal', 'I-academicjournal',
                      'B-event', 'I-event', 'B-theory', 'I-theory', 'B-award', 'I-award', 'B-misc', 'I-misc']
    music_labels = ['O', 'B-musicgenre', 'I-musicgenre', 'B-song', 'I-song', 'B-band', 'I-band', 'B-album', 'I-album',
                    'B-musicalartist', 'I-musicalartist', 'B-musicalinstrument', 'I-musicalinstrument', 'B-award',
                    'I-award', 'B-event', 'I-event', 'B-country', 'I-country', 'B-location', 'I-location',
                    'B-organisation', 'I-organisation', 'B-person', 'I-person', 'B-misc', 'I-misc']
    literature_labels = ["O", "B-book", "I-book", "B-writer", "I-writer", "B-award", "I-award", "B-poem", "I-poem",
                         "B-event", "I-event", "B-magazine", "I-magazine", "B-literarygenre", "I-literarygenre",
                         'B-country', 'I-country', "B-person", "I-person", "B-location", "I-location", 'B-organisation',
                         'I-organisation', 'B-misc', 'I-misc']
    ai_labels = ["O", "B-field", "I-field", "B-task", "I-task", "B-product", "I-product", "B-algorithm", "I-algorithm",
                 "B-researcher", "I-researcher", "B-metrics", "I-metrics", "B-programlang", "I-programlang",
                 "B-conference", "I-conference", "B-university", "I-university", "B-country", "I-country", "B-person",
                 "I-person", "B-organisation", "I-organisation", "B-location", "I-location", "B-misc", "I-misc"]

    single_wnut = ["O", "B-creative-work", "I-creative-work", "B-person", "I-person", "B-corporation", "I-corporation",
                   "B-location", "I-location", "B-group", "I-group", "B-product", "I-product"]  # 12 without "O"
    single_conll = ['O', 'B-organisation', 'I-organisation', 'B-misc', 'I-misc', 'B-person', 'I-person', 'B-location',
                    'I-location']  # 8 without "O"
    single_btc = ["O", "B-PER", "I-PER", "B-LOC", "I-LOC", "B-ORG", "I-ORG"]  # 6 without "O"
    single_drugs = ["O", "B-Drug", "I-Drug"]  # 2 without "O"

    conll_to_drugs = ['O', 'B-organisation', 'I-organisation', 'B-misc', 'I-misc', 'B-person', 'I-person', 'B-location',
                      'I-location', 'B-Drug', 'I-Drug']
    wnut_to_drugs = ["O", "B-creative-work", "I-creative-work", "B-person", "I-person", "B-corporation",
                     "I-corporation", "B-location", "I-location", "B-product", "I-product", "B-Drug", "I-Drug"]
    btc_to_drugs = ["O", "B-PER", "I-PER", "B-LOC", "I-LOC", "B-ORG", "I-ORG", "B-Drug", "I-Drug"]
    wnut_to_btc = ["O", "B-creative-work", "I-creative-work", "B-person", "I-person", "B-corporation", "I-corporation",
                   "B-location", "I-location", "B-product", "I-product", "B-PER", "I-PER", "B-LOC", "I-LOC", "B-ORG",
                   "I-ORG", "B-Drug", "I-Drug"]

    domain2labels = {"politics": politics_labels, "science": science_labels, "music": music_labels,
                     "literature": literature_labels, "ai": ai_labels}
    custom_domains = {'wnut': single_wnut, 'nutot': single_wnut, 'btc': single_btc, 'drugs': single_drugs,
                      'wiki': single_drugs, 'conll': single_conll}

    # create custom entity class set
    if tgt_dm not in domain2labels:
        if tgt_dm in custom_domains and src_dm in custom_domains:
            domain2labels[tgt_dm] = custom_domains[tgt_dm]
            domain2labels[tgt_dm].extend(custom_domains[src_dm])
            domain2labels[tgt_dm] = remove_duplicates_with_order(domain2labels[tgt_dm])  # remove duplicate entity classes
        else:
            raise Exception("Target or Source Domain not in Custom Domain Set! Fix that")
    elif params.tgt_dm not in domain2labels and params.src_dm in custom_domains and src_dm != 'conll':
        raise Exception("If you want to use a standard domain with custom source domain --> you gotta configure this.")

    return domain2labels

def split_text(token_list, label_list, cut_strat):
    dot_enc = auto_tokenizer.convert_tokens_to_ids('.')
    splitted_token_lists, splitted_label_lists = [], []
    amt_cutoff=0
    amt_total_splits=1


    if cut_strat != 1: # if cut_strat is 1 than directly cutoff long texts.
        if cut_strat == 2:
            start_at_indx = 400
            cut_until = 500
        elif cut_strat == 3:
            start_at_indx = 0#Cut into sentences
            cut_until = 0
        elif cut_strat == 4: #Ignore longer texts
            return splitted_token_lists, splitted_label_lists, len(token_list), amt_total_splits
        else:
            print('Cut Strategy: ' , str(cut_strat))
            raise ValueError
        while len(token_list) > cut_until and start_at_indx >= 0:
            try:
                cut_at = token_list.index(dot_enc, start_at_indx, 500)
                splitted_token_lists.append(token_list[:cut_at])
                splitted_label_lists.append(label_list[:cut_at])
                if cut_at < len(token_list):
                    token_list = token_list[cut_at+1:]
                    label_list = label_list[cut_at+1:]
                    amt_total_splits+=1
                else:
                    token_list = []
                    label_list = []

                if cut_strat == 2:#Reset starting point for next possible split
                    start_at_indx = 400
            except ValueError:
                logger.info("Split not found enlarging search area")
                start_at_indx = start_at_indx - 100 #If no dot is in that area --> make search are bigger or continue

    if len(token_list) > 500:
        amt_cutoff = len(token_list) - 500
        splitted_token_lists.append(token_list[:500])
        splitted_label_lists.append(label_list[:500])
    else:
        splitted_token_lists.append(token_list)
        splitted_label_lists.append(label_list)

    return splitted_token_lists, splitted_label_lists, amt_cutoff, amt_total_splits




def read_ner(datapath, tgt_dm, src_dm):
    inputs, labels = [], []
    domain2labels = get_domain2labels(tgt_dm, src_dm)
    amt_cut_off=0
    amt_tokens=0
    amt_splits=0

    with open(datapath, "r") as fr:
        token_list, label_list  = [], []

        for i, line in enumerate(fr):
            line = line.strip()
            if line == "":
                if len(token_list) > 0:
                    assert len(token_list) == len(label_list)

                    if len(token_list) > 500 or params.cut_strategy==3:
                        splitted_token_lists, splitted_label_lists, recent_cutoff, rec_splits = split_text(token_list, label_list, params.cut_strategy)
                        amt_cut_off += recent_cutoff
                        amt_splits += rec_splits



                        for cut_tokens in splitted_token_lists:
                            inputs.append([auto_tokenizer.cls_token_id] + cut_tokens + [auto_tokenizer.sep_token_id])
                        for cut_labels in splitted_label_lists:
                            labels.append([pad_token_label_id] + cut_labels + [pad_token_label_id])
                    else:
                        amt_tokens += len(token_list)
                        inputs.append([auto_tokenizer.cls_token_id] + token_list + [auto_tokenizer.sep_token_id])
                        labels.append([pad_token_label_id] + label_list + [pad_token_label_id])

                token_list, label_list = [], []
                continue

            splits = line.split("\t")

            if len(splits) != 2:
                print('i:%d, text:%s'%(i, line))

            token = splits[0]
            #Check if its wnut, since this dataset has multiple labels on the testset..
            if ( src_dm != 'wnut' and src_dm != 'nutot' and tgt_dm != 'wnut'  and tgt_dm != 'nutot' ) or "," not in splits[1]:
                label = splits[1]
            else: # If its WNUT and it has commas in the label --> prioritize product, but acutally doesn't matter much.#TODO check if I should build this in properly
                possible_labels = splits[1].split(',')
                if 'B-product' in possible_labels:
                    label = 'B-product'
                elif 'I-product' in possible_labels:
                    label = 'I-product'
                else:
                    label = possible_labels[0]


            subs_ = auto_tokenizer.tokenize(token)
            if len(subs_) > 0:
                label_list.extend([domain2labels[tgt_dm].index(label)] + [pad_token_label_id] * (len(subs_) - 1))
                token_list.extend(auto_tokenizer.convert_tokens_to_ids(subs_))
            else:
                print("length of subwords for %s is zero; its label is %s" % (token, label))

    logger.info('Read total of %d tokens and cutoff %d tokens for file %s and split/cut at least into %d times'%(amt_tokens, amt_cut_off, datapath, amt_splits))
    return inputs, labels


def read_ner_for_bilstm(datapath, tgt_dm, vocab):
    domain2labels = get_domain2labels(tgt_dm, 'Error')
    inputs, labels = [], []
    with open(datapath, "r") as fr:
        token_list, label_list = [], []
        for i, line in enumerate(fr):
            line = line.strip()
            if line == "":
                if len(token_list) > 0:
                    assert len(token_list) == len(label_list)
                    inputs.append(token_list)
                    labels.append(label_list)

                token_list, label_list = [], []
                continue

            splits = line.split("\t")
            token = splits[0]
            label = splits[1]

            token_list.append(vocab.word2index[token])
            label_list.append(domain2labels[tgt_dm].index(label))

    return inputs, labels


class Dataset(data.Dataset):
    def __init__(self, inputs, labels):
        self.X = inputs
        self.y = labels

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return len(self.X)


PAD_INDEX = 0


class Vocab():
    def __init__(self):
        self.word2index = {"PAD": PAD_INDEX}
        self.index2word = {PAD_INDEX: "PAD"}
        self.n_words = 1

    def index_words(self, word_list):
        for word in word_list:
            if word not in self.word2index:
                self.word2index[word] = self.n_words
                self.index2word[self.n_words] = word
                self.n_words += 1


def get_vocab(path):
    vocabulary = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            vocabulary.append(line)
    return vocabulary


def collate_fn(data):
    X, y = zip(*data)
    lengths = [len(bs_x) for bs_x in X]
    max_lengths = max(lengths)
    padded_seqs = torch.LongTensor(len(X), max_lengths).fill_(auto_tokenizer.pad_token_id)
    padded_y = torch.LongTensor(len(X), max_lengths).fill_(pad_token_label_id)
    for i, (seq, y_) in enumerate(zip(X, y)):
        length = lengths[i]
        padded_seqs[i, :length] = torch.LongTensor(seq)
        padded_y[i, :length] = torch.LongTensor(y_)

    return padded_seqs, padded_y


def collate_fn_for_bilstm(data):
    X, y = zip(*data)
    lengths = [len(bs_x) for bs_x in X]
    max_lengths = max(lengths)
    padded_seqs = torch.LongTensor(len(X), max_lengths).fill_(PAD_INDEX)
    for i, seq in enumerate(X):
        length = lengths[i]
        padded_seqs[i, :length] = torch.LongTensor(seq)

    lengths = torch.LongTensor(lengths)
    return padded_seqs, lengths, y


def get_dataloader_for_bilstmtagger(params):
    vocab_src = get_vocab("ner_data/conll2003/vocab.txt")
    vocab_tgt = get_vocab("ner_data/%s/vocab.txt" % params.tgt_dm)
    vocab = Vocab()
    vocab.index_words(vocab_src)
    vocab.index_words(vocab_tgt)

    logger.info("Load training set data ...")
    conll_inputs_train, conll_labels_train = read_ner_for_bilstm("ner_data/conll2003/train.txt", params.tgt_dm, vocab)
    inputs_train, labels_train = read_ner_for_bilstm("ner_data/%s/train.txt" % params.tgt_dm, params.tgt_dm, vocab)
    inputs_train = inputs_train * 10 + conll_inputs_train
    labels_train = labels_train * 10 + conll_labels_train

    logger.info("Load dev set data ...")
    inputs_dev, labels_dev = read_ner_for_bilstm("ner_data/%s/dev.txt" % params.tgt_dm, params.tgt_dm, vocab)

    logger.info("Load test set data ...")
    inputs_test, labels_test = read_ner_for_bilstm("ner_data/%s/test.txt" % params.tgt_dm, params.tgt_dm, vocab)

    logger.info("train size: %d; dev size %d; test size: %d;" % (len(inputs_train), len(inputs_dev), len(inputs_test)))

    dataset_train = Dataset(inputs_train, labels_train)
    dataset_dev = Dataset(inputs_dev, labels_dev)
    dataset_test = Dataset(inputs_test, labels_test)

    dataloader_train = DataLoader(dataset=dataset_train, batch_size=params.batch_size, shuffle=True,
                                  collate_fn=collate_fn_for_bilstm)
    dataloader_dev = DataLoader(dataset=dataset_dev, batch_size=params.batch_size, shuffle=False,
                                collate_fn=collate_fn_for_bilstm)
    dataloader_test = DataLoader(dataset=dataset_test, batch_size=params.batch_size, shuffle=False,
                                 collate_fn=collate_fn_for_bilstm)

    return dataloader_train, dataloader_dev, dataloader_test, vocab


def load_corpus(tgt_dm):
    print("Loading corpus ...")
    data_path = "enwiki_corpus/%s_removebracket.tok" % tgt_dm
    sent_list = []
    with open(data_path, "r") as fr:
        for i, line in tqdm(enumerate(fr)):
            line = line.strip()
            sent_list.append(line)
    return sent_list


def get_dataloader(params):
    logger.info("Load training set data")
    inputs_train, labels_train = read_ner("ner_data/%s/train.txt" % params.tgt_dm, params.tgt_dm, params.src_dm)
    domain2labels = get_domain2labels(params.tgt_dm, params.src_dm)

    if params.n_samples != -1:
        logger.info("Few-shot on %d samples" % params.n_samples)
        inputs_train = inputs_train[:params.n_samples]
        labels_train = labels_train[:params.n_samples]
    logger.info("Load development set data")
    inputs_dev, labels_dev = read_ner("ner_data/%s/dev.txt" % params.tgt_dm, params.tgt_dm, params.src_dm)
    logger.info("Load test set data")
    inputs_test, labels_test = read_ner("ner_data/%s/test.txt" % params.tgt_dm, params.tgt_dm, params.src_dm)

    logger.info("label distribution for training set")
    label_distri_train = {}
    count_tok_train = 0
    for label_seq in labels_train:
        for label in label_seq:
            if label != pad_token_label_id:
                label_name = domain2labels[params.tgt_dm][label]
                if "B" in label_name:
                    count_tok_train += 1
                    label_name = label_name.split("-")[1]
                    if label_name not in label_distri_train:
                        label_distri_train[label_name] = 1
                    else:
                        label_distri_train[label_name] += 1
    print(label_distri_train)
    for key in label_distri_train:
        label_distri_train[key] /= count_tok_train
    logger.info(label_distri_train)

    logger.info("label distribution for dev set")
    label_distri_dev = {}
    count_tok_test = 0
    for label_seq in labels_dev:
        for label in label_seq:
            if label != pad_token_label_id:
                label_name = domain2labels[params.tgt_dm][label]
                if "B" in label_name:
                    count_tok_test += 1
                    label_name = label_name.split("-")[1]
                    if label_name not in label_distri_dev:
                        label_distri_dev[label_name] = 1
                    else:
                        label_distri_dev[label_name] += 1
    print(label_distri_dev)
    for key in label_distri_dev:
        label_distri_dev[key] /= count_tok_test
    logger.info(label_distri_dev)

    logger.info("label distribution for test set")
    label_distri_test = {}
    count_tok_test = 0
    for label_seq in labels_test:
        for label in label_seq:
            if label != pad_token_label_id:
                label_name = domain2labels[params.tgt_dm][label]
                if "B" in label_name:
                    count_tok_test += 1
                    label_name = label_name.split("-")[1]
                    if label_name not in label_distri_test:
                        label_distri_test[label_name] = 1
                    else:
                        label_distri_test[label_name] += 1
    print(label_distri_test)
    for key in label_distri_test:
        label_distri_test[key] /= count_tok_test
    logger.info(label_distri_test)

    if params.conll and params.joint: #No worries, this is only done when training is executed in a joint manner. Furthermore our dataset are nearly equal so balancing is not so important.
        conll_inputs_train, conll_labels_train = read_ner("ner_data/conll2003/train.txt", params.tgt_dm, params.src_dm)
        inputs_train = inputs_train * 50  # augment the target domain data to balance the source and target domain data
        labels_train = labels_train * 50
        inputs_train = inputs_train + conll_inputs_train
        labels_train = labels_train + conll_labels_train

    logger.info("train size: %d; dev size %d; test size: %d;" % (len(inputs_train), len(inputs_dev), len(inputs_test)))

    dataset_train = Dataset(inputs_train, labels_train)
    dataset_dev = Dataset(inputs_dev, labels_dev)
    dataset_test = Dataset(inputs_test, labels_test)

    dataloader_train = DataLoader(dataset=dataset_train, batch_size=params.batch_size, shuffle=True,
                                  collate_fn=collate_fn)
    dataloader_dev = DataLoader(dataset=dataset_dev, batch_size=params.batch_size, shuffle=False, collate_fn=collate_fn)
    dataloader_test = DataLoader(dataset=dataset_test, batch_size=params.batch_size, shuffle=False,
                                 collate_fn=collate_fn)

    return dataloader_train, dataloader_dev, dataloader_test


def get_conll2003_dataloader(batch_size, params):
    inputs_train, labels_train = read_ner("ner_data/conll2003/train.txt", params.tgt_dm, params.src_dm)
    inputs_dev, labels_dev = read_ner("ner_data/conll2003/dev.txt", params.tgt_dm, params.src_dm)
    inputs_test, labels_test = read_ner("ner_data/conll2003/test.txt", params.tgt_dm, params.src_dm)

    logger.info("conll2003 dataset: train size: %d; dev size %d; test size: %d" % (
    len(inputs_train), len(inputs_dev), len(inputs_test)))

    dataset_train = Dataset(inputs_train, labels_train)
    dataset_dev = Dataset(inputs_dev, labels_dev)
    dataset_test = Dataset(inputs_test, labels_test)

    dataloader_train = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    dataloader_dev = DataLoader(dataset=dataset_dev, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    dataloader_test = DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return dataloader_train, dataloader_dev, dataloader_test


def get_btc_dataloader(batch_size, params):
    # According to https://github.com/GateNLP/broad_twitter_corpus
    # Training: everything else
    inputs_train1, labels_train1 = read_ner("ner_data/btc/a.conll", params.tgt_dm, params.src_dm)
    inputs_train2, labels_train2 = read_ner("ner_data/btc/b.conll", params.tgt_dm, params.src_dm)
    inputs_train3, labels_train3 = read_ner("ner_data/btc/e.conll", params.tgt_dm, params.src_dm)
    inputs_train4, labels_train4 = read_ner("ner_data/btc/g.conll", params.tgt_dm, params.src_dm)
    inputs_train5, labels_train5 = read_ner("ner_data/btc/h.conll", params.tgt_dm, params.src_dm)

    # Development: second half of Section H
    inputs_dev = inputs_train5[math.floor( len(inputs_train5) / 2):]
    labels_dev = labels_train5[math.floor( len(labels_train5) / 2):]
    inputs_train5 = inputs_train5[:math.floor( len(inputs_train5) / 2)]
    labels_train5 = labels_train5[:math.floor( len(labels_train5) / 2)]

    # Test: Section F
    inputs_test, labels_test = read_ner("ner_data/btc/f.conll", params.tgt_dm, params.src_dm)

    # Aggregate Train sets
    inputs_train = inputs_train1 + inputs_train2 + inputs_train3 + inputs_train4 + inputs_train5
    labels_train = labels_train1 + labels_train2 + labels_train3 + labels_train4 + labels_train5

    logger.info("Broad Twitter Corpus dataset: train size: %d; dev size %d; test size: %d" % (
        len(inputs_train), len(inputs_dev), len(inputs_test)))

    dataset_train = Dataset(inputs_train, labels_train)
    dataset_dev = Dataset(inputs_dev, labels_dev)
    dataset_test = Dataset(inputs_test, labels_test)

    dataloader_train = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    dataloader_dev = DataLoader(dataset=dataset_dev, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    dataloader_test = DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return dataloader_train, dataloader_dev, dataloader_test


def get_wnut_dataloader(batch_size, params):
    inputs_train, labels_train = read_ner("ner_data/wnut/train.txt", params.tgt_dm, params.src_dm)
    inputs_dev, labels_dev = read_ner("ner_data/wnut/dev.txt", params.tgt_dm, params.src_dm)
    inputs_test, labels_test = read_ner("ner_data/wnut/test.txt", params.tgt_dm, params.src_dm)

    logger.info("W-NUT 2017 dataset: train size: %d; dev size %d; test size: %d" % (
        len(inputs_train), len(inputs_dev), len(inputs_test)))

    dataset_train = Dataset(inputs_train, labels_train)
    dataset_dev = Dataset(inputs_dev, labels_dev)
    dataset_test = Dataset(inputs_test, labels_test)

    dataloader_train = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    dataloader_dev = DataLoader(dataset=dataset_dev, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    dataloader_test = DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return dataloader_train, dataloader_dev, dataloader_test


def get_generic_source_dataloader(batch_size, params):
    inputs_train, labels_train = read_ner("ner_data/%s/train.txt" % params.src_dm, params.tgt_dm, params.src_dm)
    inputs_dev, labels_dev = read_ner("ner_data/%s/dev.txt" % params.src_dm, params.tgt_dm, params.src_dm)
    inputs_test, labels_test = read_ner("ner_data/%s/test.txt" % params.src_dm, params.tgt_dm, params.src_dm)

    logger.info("%s dataset: train size: %d; dev size %d; test size: %d" % (params.src_dm,
                                                                            len(inputs_train), len(inputs_dev),
                                                                            len(inputs_test)))

    dataset_train = Dataset(inputs_train, labels_train)
    dataset_dev = Dataset(inputs_dev, labels_dev)
    dataset_test = Dataset(inputs_test, labels_test)

    dataloader_train = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    dataloader_dev = DataLoader(dataset=dataset_dev, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    dataloader_test = DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return dataloader_train, dataloader_dev, dataloader_test


def get_wiki_dataloader(batch_size, params):
    inputs_train, labels_train = read_ner("ner_data/wiki/train.txt", params.tgt_dm, params.src_dm)
    inputs_dev, labels_dev = read_ner("ner_data/wiki/dev.txt", params.tgt_dm, params.src_dm)
    inputs_test, labels_test = read_ner("ner_data/wiki/test.txt", params.tgt_dm, params.src_dm)

    logger.info("Wikipedia distantly supervised dataset: train size: %d; dev size %d; test size: %d" % (
        len(inputs_train), len(inputs_dev), len(inputs_test)))

    dataset_train = Dataset(inputs_train, labels_train)
    dataset_dev = Dataset(inputs_dev, labels_dev)
    dataset_test = Dataset(inputs_test, labels_test)

    dataloader_train = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    dataloader_dev = DataLoader(dataset=dataset_dev, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    dataloader_test = DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return dataloader_train, dataloader_dev, dataloader_test


if __name__ == "__main__":
    read_ner("../ner_data/drugs/train.txt", "drugs", "wnut")
