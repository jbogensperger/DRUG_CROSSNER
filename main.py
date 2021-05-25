import os

#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#CUos.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,"  # GPUs

from src.config import get_params
from src.utils import init_experiment
from src.dataloader import get_dataloader, get_conll2003_dataloader, get_dataloader_for_bilstmtagger, \
    get_wnut_dataloader, get_btc_dataloader, get_generic_source_dataloader
from src.trainer import BaseTrainer
from src.model import BertTagger, BiLSTMTagger
from src.coach.dataloader import get_dataloader_for_coach
from src.coach.model import EntityPredictor
from src.coach.trainer import CoachTrainer

import torch
import numpy as np
from tqdm import tqdm
import random


def random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def train(params):
    # initialize experiment
    logger = init_experiment(params, logger_filename=params.logger_filename)
    
    if params.bilstm:
        # dataloader
        dataloader_train, dataloader_dev, dataloader_test, vocab = get_dataloader_for_bilstmtagger(params)
        # bilstm-crf model
        model = BiLSTMTagger(params, vocab)
        model.cuda()
        # trainer
        trainer = BaseTrainer(params, model)
    elif params.coach:
        # dataloader
        dataloader_train, dataloader_dev, dataloader_test, vocab = get_dataloader_for_coach(params)
        # coach model
        binary_tagger = BiLSTMTagger(params, vocab)
        entity_predictor = EntityPredictor(params)
        binary_tagger.cuda()
        entity_predictor.cuda()
        # trainer
        trainer = CoachTrainer(params, binary_tagger, entity_predictor)
    else:
        # dataloader
        dataloader_train, dataloader_dev, dataloader_test = get_dataloader(params)
        # BERT-based NER Tagger
        model = BertTagger(params)
        model.cuda()
        # trainer
        trainer = BaseTrainer(params, model)

    if params.conll and not params.joint:
        conll_trainloader, conll_devloader, conll_testloader = get_conll2003_dataloader(params.batch_size, params)
        trainer.train_source_domain(conll_trainloader, conll_devloader, conll_testloader, params.tgt_dm, params.src_dm, "CONLL")

    elif params.wnut and not params.joint:
        wnut_trainloader, wnut_devloader, wnut_testloader = get_wnut_dataloader(params.batch_size, params)
        trainer.train_source_domain(wnut_trainloader, wnut_devloader, wnut_testloader, params.tgt_dm, params.src_dm, "WNUT")

    elif ( params.wnut_nutot or params.wiki or params.nutot ) and not params.joint:
        generic_trainloader, generic_devloader, generic_testloader = get_generic_source_dataloader(params.batch_size, params)
        trainer.train_source_domain(generic_trainloader, generic_devloader, generic_testloader, params.tgt_dm, params.src_dm, params.src_dm)

    elif params.btc and not params.joint:
        btc_trainloader, btc_devloader, btc_testloader = get_btc_dataloader(params.batch_size, params)
        trainer.train_source_domain(btc_trainloader, btc_devloader, btc_testloader, params.tgt_dm, params.src_dm, "BTC")


    no_improvement_num = 0
    best_f1 = 0
    logger.info("*********************************************")
    logger.info("*******   Training on target domain   *******")
    logger.info("*********************************************")
    for e in range(params.epoch):
        logger.info("============== epoch %d ==============" % e)
        
        pbar = tqdm(enumerate(dataloader_train), total=len(dataloader_train))
        if params.bilstm:
            loss_list = []
            for i, (X, lengths, y) in pbar:
                X, lengths = X.cuda(), lengths.cuda()
                loss = trainer.train_step_for_bilstm(X, lengths, y)
                loss_list.append(loss)
                pbar.set_description("(Epoch {}) LOSS:{:.4f}".format(e, np.mean(loss_list)))

            logger.info("Finish training epoch %d. loss: %.4f" % (e, np.mean(loss_list)))

        elif params.coach:
            loss_bin_list, loss_entity_list = [], []
            for i, (X, lengths, y_bin, y_final) in pbar:
                X, lengths = X.cuda(), lengths.cuda()
                loss_bin, loss_entityname = trainer.train_step(X, lengths, y_bin, y_final)
                loss_bin_list.append(loss_bin)
                loss_entity_list.append(loss_entityname)
                pbar.set_description("(Epoch {}) LOSS BIN:{:.4f}; LOSS ENTITY:{:.4f}".format(e, np.mean(loss_bin_list), np.mean(loss_entity_list)))
            
            logger.info("Finish training epoch %d. loss_bin: %.4f. loss_entity: %.4f" % (e, np.mean(loss_bin_list), np.mean(loss_entity_list)))

        else:
            loss_list = []
            for i, (X, y) in pbar:
                X, y = X.cuda(), y.cuda()
                loss = trainer.train_step(X, y)
                loss_list.append(loss)
                pbar.set_description("(Epoch {}) LOSS:{:.4f}".format(e, np.mean(loss_list)))

            logger.info("Finish training epoch %d. loss: %.4f" % (e, np.mean(loss_list)))

        logger.info("============== Evaluate epoch %d on Train Set ==============" % e)
        f1_train, prec_train, rec_train, per_class_train = trainer.evaluate(dataloader_train, params.tgt_dm, params.src_dm, use_bilstm=params.bilstm)
        logger.info("Evaluate on Train Set.F1: %.4f. Prec.: %.4f. Rec.: %.4f." % (f1_train, prec_train, rec_train))

        logger.info("============== Evaluate epoch %d on Dev Set ==============" % e)
        f1_dev, prec_dev, rec_dev, per_class_dev = trainer.evaluate(dataloader_dev, params.tgt_dm, params.src_dm, use_bilstm=params.bilstm)
        logger.info("Evaluate on Dev Set. F1: %.4f. Prec.: %.4f. Rec.: %.4f." % (f1_dev, prec_dev, rec_dev))



        if f1_dev > best_f1:
            logger.info("Found better model!!")
            best_f1 = f1_dev
            no_improvement_num = 0
            # trainer.save_model()

            logger.info("============== Evaluate epoch %d on Test Set ==============" % e)
            f1_test, prec_test, rec_test, per_class_test = trainer.evaluate(dataloader_test, params.tgt_dm, params.src_dm, use_bilstm=params.bilstm)
            #Per Class DiCT has entries: tp, fp, fn, prec, rec, fscore contained for each entity type/class
            logger.info("Evaluate on Test Set. F1: %.4f. Prec.: %.4f. Rec.: %.4f." %(f1_test, prec_test, rec_test))


        else:
            no_improvement_num += 1
            logger.info("No better model found (%d/%d)" % (no_improvement_num, params.early_stop))

        if no_improvement_num >= params.early_stop:
            break

    #Save Model if a dump path is provided:
    if params.save:
        trainer.save_model()


if __name__ == "__main__":
    params = get_params()

    random_seed(params.seed)
    train(params)
