# -*- coding: utf-8 -*-
import random
import argparse
import datetime
import numpy as np
import torch
from torch.optim.sgd import SGD as TorchSGD
from torch.utils.data.dataloader import DataLoader as TorchDataLoader
from utils.data import TwoStreamBatchSampler

import GLOB as glob
import datasources
import datasets
import models as modelClass
from models.base.ema import ModelEMA

from utils.base.log import Logger
from utils.base.comm import CommUtils as comm
from utils.process import ProcessUtils as proc
from utils.parameters import consWeight_increase
from utils.losses import ClassLoss, ClassFixLoss, ClassDistLoss, AvgCounter
from sklearn.metrics import precision_score


def main(args):
    allTM = datetime.datetime.now()
    logger = glob.getValue("logger")
    logger.print("L1", "=> {}, start".format(args.experiment))
    args.start_epoch = 0
    args.best_acc = [-1.]
    args.best_epoch = [0]

    # region 1. Initialize
    # Data loading
    dataSource = datasources.__dict__[args.dataset]()
    labeled_idxs, unlabeled_idxs, labeled_idxs_exp, unlabeled_idxs_exp, mean, std, validDS = dataSource.getSemiData(num_labeled=args.trainCount_labeled, batch_size=args.trainBS, mu=args.mu, iter_num=args.iter_num)
    args.num_classes, args.imgType, args.inpRes, args.outRes = dataSource.num_classes, dataSource.imgType, dataSource.inpRes, dataSource.outRes

    # Model initialize
    model_pNum = 0
    model = modelClass.__dict__["ClassModel"](modelType=args.model, num_classes=args.num_classes, mode=args.feature_mode, dataset=args.dataset).to(args.device)
    model_ema = ModelEMA(model, args.ema_decay, args)

    no_decay = ['bias', 'bn']
    grouped_parameters = [{'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
                          {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    optim = TorchSGD(grouped_parameters, lr=args.lr, momentum=args.momentum, nesterov=args.nesterov)

    model_pNum += (sum(p.numel() for p in model.parameters()))
    logc = "=> initialized FixMatch ({}) Structure (params: {})".format(args.model, format(model_pNum / (1024**2), ".2f"))
    logger.print("L1", logc)

    # Dataloader initialize
    trainDS = datasets.__dict__[args.dataset](dataSource.root, mean, std, labeled_idxs, augCount=args.augNum, **vars(args))
    batchSampler = TwoStreamBatchSampler(unlabeled_idxs_exp, labeled_idxs_exp, (args.mu+1)*args.trainBS, args.trainBS)
    trainLoader = TorchDataLoader(trainDS, batch_sampler=batchSampler, pin_memory=True, drop_last=False)
    validLoader = TorchDataLoader(validDS, batch_size=args.inferBS, shuffle=True, pin_memory=True, drop_last=False)
    logger.print("L1", "=> initialized {} Dataloaders".format(args.dataset))
    # endregion

    # region 2. Iteration
    logger.print("L1", "=> training start")
    for epo in range(args.epochs):
        epoTM = datetime.datetime.now()
        args.epo = epo

        # region 2.1 update dynamic parameters
        args.consWeight = consWeight_increase(epo, args)
        # endregion

        # region 2.2 model training and validating
        startTM = datetime.datetime.now()
        total_loss, clc_loss, fix_loss, mtc_loss, fix_mask, mtc_mask = train(trainLoader, model, model_ema, optim, args)
        logger.print("L3", "model training finished...", start=startTM)

        startTM = datetime.datetime.now()
        predsArray, accs = validate(validLoader, model_ema, args)
        logger.print("L3", "model validating finished...", start=startTM)
        # endregion

        # region 2.3 model selection & storage
        startTM = datetime.datetime.now()
        # model selection
        is_best = accs[0] > args.best_acc
        if is_best:
            args.best_epoch = epo
            args.best_acc = accs[0]
        # model storage
        checkpoint = {"current_epoch": args.epo, "best_acc": args.best_acc, "best_epoch": args.best_epoch,
                      "model": args.model, "feature_mode": args.feature_mode,
                      "model_state": model.state_dict(), "optim_state": optim.state_dict(),
                      "model_ema_state": (model_ema.ema.module if hasattr(model_ema.ema, "module") else model_ema.ema).state_dict()}
        comm.ckpt_save(checkpoint, is_best, ckptPath="{}/ckpts".format(args.basePath))
        logger.print("L3", "model storage finished...", start=startTM)
        # endregion

        # region 2.4 log storage
        startTM = datetime.datetime.now()
        # Initialization parameter storage
        if epo == args.start_epoch:
            save_args = vars(args).copy()
            save_args.pop("device")
            comm.json_save(save_args, "{}/logs/args.json".format(args.basePath), isCover=True)

        # Log data storage
        log_data = {"total_loss": total_loss, "clc_loss": clc_loss, "fix_loss": fix_loss, "mtc_loss": mtc_loss, "fix_mask": fix_mask, "mtc_mask": mtc_mask,
                    "AP": accs[0], "mAP": accs[1]}
        comm.json_save(log_data, "{}/logs/logData/logData_{}.json".format(args.basePath, epo+1), isCover=True)

        # Pseudo-labels data storage
        pseudo_data = {"predsArray": predsArray}
        comm.json_save(pseudo_data, "{}/logs/pseudoData/pseudoData_{}.json".format(args.basePath, epo+1), isCover=True)
        logger.print("L3", "log storage finished...", start=startTM)
        # endregion

        # region 2.5 output result
        # Training performance
        fmtc = "[{}/{} | lr: {} | fixW: {}, consW: {}, fix_mask: {}, mtc_mask: {}] total_loss: {}, clc_loss: {}, fix_loss: {}, mtc_loss: {}"
        logc = fmtc.format(format(epo + 1, "3d"), format(args.epochs, "3d"),
                           format(optim.state_dict()['param_groups'][0]['lr'], ".3f"),
                           format(args.fixWeight, ".2f"), format(args.consWeight, ".2f"), format(fix_mask, ".3f"), format(mtc_mask, ".3f"),
                           format(total_loss, ".5f"), format(clc_loss, ".5f"), format(fix_loss, ".10f"), format(mtc_loss, ".10f"))
        logger.print("L1", logc)

        # Validating performance
        fmtc = "[{}/{}] best AP: {} (epo: {}) | AP: {}, mAP: {}"
        logc = fmtc.format(format(epo + 1, "3d"), format(args.epochs, "3d"),
                           format(args.best_acc, ".5f"), format(args.best_epoch + 1, "3d"),
                           format(accs[0], ".5f"), format(accs[1], ".5f"))
        logger.print("L1", logc)

        # Epoch line
        time_interval = logger._interval_format(seconds=(datetime.datetime.now() - epoTM).seconds*(args.epochs - (epo+1)))
        fmtc = "[{}/{} | {}] ---------- ---------- ---------- ---------- ---------- ---------- ----------"
        logc = fmtc.format(format(epo + 1, "3d"), format(args.epochs, "3d"), time_interval)
        logger.print("L1", logc, start=epoTM)
        # endregion
    # endregion

    logger.print("L1", "[{}, all executing finished...]".format(args.experiment), start=allTM)


def interleave(x, size):
    s = list(x.shape)
    return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def de_interleave(x, size):
    s = list(x.shape)
    return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def train(trainLoader, model, model_ema, optim, args):
    # region 1. Preparation
    logger = glob.getValue("logger")
    total_counter = AvgCounter()
    clc_counter = AvgCounter()
    fix_counter = AvgCounter()
    fix_mask_counter = AvgCounter()
    mtc_counter = AvgCounter()
    mtc_mask_counter = AvgCounter()
    class_criterion = ClassLoss().to(args.device)
    fix_criterion = ClassFixLoss(scoreThr= args.scoreThr).to(args.device)
    consistency_criterion = ClassDistLoss(scoreThr= args.scoreThr).to(args.device)
    model.train()
    model_ema.ema.train()
    # endregion

    # region 2. Training
    for bat, (imgs_labeled, imgs_strong, imgs_weak, labels, meta) in enumerate(trainLoader):
        optim.zero_grad()

        # region 2.1 Data organizing
        imgs_labeled = [torch.stack([img for iIdx, img in enumerate(imgs_labeled[aIdx]) if meta['islabeled'][iIdx] > 0], dim=0) for aIdx in range(len(imgs_labeled))]
        imgs_labeled = [proc.setVariable(img, args.device) for img in imgs_labeled]
        labels = torch.stack([label for iIdx, label in enumerate(labels) if meta['islabeled'][iIdx] > 0], dim=0)
        labels = proc.setVariable(labels, args.device, False)

        imgs_strong = [torch.stack([img for iIdx, img in enumerate(imgs_strong[aIdx]) if meta['islabeled'][iIdx] == 0], dim=0) for aIdx in range(len(imgs_strong))]
        imgs_strong = [proc.setVariable(img, args.device) for img in imgs_strong]

        imgs_weak = [torch.stack([img for iIdx, img in enumerate(imgs_weak[aIdx]) if meta['islabeled'][iIdx] == 0], dim=0) for aIdx in range(len(imgs_weak))]
        imgs_weak = [proc.setVariable(img, args.device) for img in imgs_weak]
        # endregion

        # region 2.2 model forward
        outs_labeled, outs_weak, outs_strong = [], [], []
        for aIdx in range(len(imgs_labeled)):
            bs = imgs_labeled[aIdx].shape[0]
            if args.useInterleave:
                inputs_cat = interleave(torch.cat((imgs_labeled[aIdx], imgs_weak[aIdx], imgs_strong[aIdx])), 2*args.mu+1).to(args.device)
            else:
                inputs_cat = torch.cat((imgs_labeled[aIdx], imgs_weak[aIdx], imgs_strong[aIdx])).to(args.device)
            outs_cat = model(inputs_cat) if args.feature_mode == "default" else model(inputs_cat)[0]  # outs: [bs, k]; labels: [bs]
            if args.useInterleave:
                outs_cat = de_interleave(outs_cat, 2 * args.mu + 1)
            outs_labeled.append(outs_cat[:bs])
            out_weak, out_strong = outs_cat[bs:].chunk(2)
            outs_weak.append(out_weak)
            outs_strong.append(out_strong)
        outs_labeled = torch.stack(outs_labeled, dim=0)
        outs_weak = torch.stack(outs_weak, dim=0)
        outs_strong = torch.stack(outs_strong, dim=0)

        if args.useConsistency:
            outs_weak_ema = []
            with torch.no_grad():
                for aIdx in range(len(imgs_labeled)):
                        out_weak = model_ema.ema(imgs_weak[aIdx]) if args.feature_mode == "default" else model_ema.ema(imgs_weak[aIdx])[0]
                        outs_weak_ema.append(out_weak)
            outs_weak_ema = torch.stack(outs_weak_ema, dim=0)
        # endregion

        # region 2.3 classification constraint
        clc_sum, clc_count = 0., 0
        for aIdx in range(len(outs_labeled)):  # [augNum, bs, k]
            loss, n = class_criterion(outs_labeled[aIdx], labels)
            clc_sum += loss
            clc_count += n
        clc_loss = args.classWeight * ((clc_sum / clc_count) if clc_count > 0 else clc_sum)
        clc_counter.update(clc_loss.item(), clc_count)
        # endregion

        # region 2.4 fix_sum, fix_count
        fix_sum, fix_count = 0., 0
        mask_sum, mask_num, mask_lens = 0., 0, 0
        for aIdx in range(len(outs_weak)):  # [augNum, bs, k]
            loss, n, mask_value, mask_len = fix_criterion(outs_strong[aIdx], outs_weak[aIdx].detach())
            fix_sum += loss
            fix_count += n
            mask_sum += mask_value.item()
            mask_lens += mask_len
            mask_num += 1
        fix_loss = args.fixWeight * ((fix_sum / fix_count) if fix_count > 0 else fix_sum)
        fix_counter.update(fix_loss.item(), fix_count)
        fix_mask_counter.update(mask_sum/mask_num, mask_num)
        # endregion

        # region 2.5 mean-teacher consistency constraint
        if args.useConsistency:
            mtc_sum, mtc_count = 0., 0
            mask_sum, mask_num, mask_lens = 0., 0, 0
            for aIdx in range(len(outs_weak_ema)):  # [augNum, bs, k]
                loss, n, mask_value, mask_len = consistency_criterion(outs_strong[aIdx], outs_weak_ema[aIdx].detach())
                mtc_sum += loss
                mtc_count += n
                mask_sum += mask_value.item()
                mask_lens += mask_len
                mask_num += 1
            mtc_loss = args.consWeight * ((mtc_sum / mtc_count) if mtc_count > 0 else mtc_sum)
            mtc_counter.update(mtc_loss.item(), mtc_count)
            mtc_mask_counter.update(mask_sum/mask_num, mask_num)
        else:
            mtc_loss = 0.
            mtc_counter.update(0., 1)
            mtc_mask_counter.update(0., 1)
        # endregion

        # region 2.6 calculate total loss & update model
        total_loss = clc_loss + fix_loss + mtc_loss
        total_counter.update(total_loss.item(), 1)
        total_loss.backward()
        optim.step()
        model_ema.update(model)
        # endregion

        # region 2.9 clearing the GPU Cache
        del outs_labeled, outs_weak, outs_strong
        # endregion
    # endregion
    return total_counter.avg, clc_counter.avg, fix_counter.avg, mtc_counter.avg, fix_mask_counter.avg, mtc_mask_counter.avg


def validate(validLoader, model_ema, args):
    # region 1. Preparation
    predsArray, labelsArray = [], []
    model_ema.ema.eval()
    # endregion

    # region 2. Validating
    with torch.no_grad():
        for bat, (imgs, labels) in enumerate(validLoader):
            # region 2.1 data organize
            imgs = proc.setVariable(imgs, args.device, False)
            labels = proc.setVariable(labels, args.device, False)
            labelsArray += labels.clone().cpu().data.numpy().tolist()
            # endregion

            # region 2.2 model predict
            # model forward
            outs_ema = model_ema.ema(imgs) if args.feature_mode == "default" else model_ema.ema(imgs)[0]

            # prediction
            _, preds_ema = torch.max(outs_ema.data, -1)
            predsArray += preds_ema.clone().cpu().data.numpy().tolist()
            # endregion

            # region 2.3 clearing the GPU Cache
            del outs_ema
            # endregion
    # endregion

    # region 3 calculate the accuracy
    validArray = [precision_score(labelsArray, predsArray, average="micro"),
                  precision_score(labelsArray, predsArray, average="macro")]
    # endregion
    return predsArray, validArray


def setArgs(args, params):
    dict_args = vars(args)
    if params is not None:
        for key in params.keys():
            if key in dict_args.keys():
                dict_args[key] = params[key]
    for key in dict_args.keys():
        if dict_args[key] == "True": dict_args[key] = True
        if dict_args[key] == "False": dict_args[key] = False
    return argparse.Namespace(**dict_args)


def exec(expMark, params=None):
    random_seed = 1388
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    args = initArgs(params)
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.experiment = "{}({}_{})_{}_{}".format(args.dataset, args.trainCount, args.trainCount_labeled, expMark, datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
    args.basePath = "{}/{}".format(glob.expr, args.experiment)
    glob.setValue("logger", Logger(args.experiment, consoleLevel="L1"))
    main(args)


def initArgs(params=None):
    # region 1. Parameters
    parser = argparse.ArgumentParser(description="FixMatch Training")

    # Model setting
    parser.add_argument("--model", default="WideResNet", choices=["WideResNet"])
    parser.add_argument("--feature_mode", default="AvgPool", choices=["AvgPool"])
    parser.add_argument("--augNum", default=1, type=int)

    # Dataset setting
    parser.add_argument("--dataset", default="CIFAR10", choices=["CIFAR10", "CIFAR100", "SVHN"], help='dataset name')
    parser.add_argument("--trainCount", default=50000, type=int, help='number of total training data')
    parser.add_argument("--trainCount_labeled", default=1000, type=int, help='number of labeled data')
    parser.add_argument("--validCount", default=10000, type=int, help='number of validating data')

    # Training strategy
    parser.add_argument("--epochs", default=50, type=int, help="the number of total epochs"),
    parser.add_argument("--iter_num", default=1024, type=int, help="the number of total epochs")
    parser.add_argument("--trainBS", default=32, type=int, help="the batchSize of training")
    parser.add_argument("--mu", default=7, type=int)
    parser.add_argument("--inferBS", default=512, type=int, help="the batchSize of infering")
    parser.add_argument("--lr", default=0.03, type=float, help="initial learning rate")
    parser.add_argument("--momentum", default=0.9, type=float, help="momentum value")
    parser.add_argument('--nesterov', default="True", help='use nesterov momentum')
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay')
    parser.add_argument("--ema_decay", default=0.999, type=float, help="ema variable decay rate (default: 0.999)")

    # Hyper-parameter
    parser.add_argument("--classWeight", default=1.0, type=float)

    parser.add_argument('--useConsistency', default="True", help='use EMA model in training')
    parser.add_argument("--consWeight_max", default=1.0, type=float)
    parser.add_argument("--consWeight_min", default=0.0, type=float)
    parser.add_argument("--consWeight_start", default=5, type=float)
    parser.add_argument("--consWeight_rampup", default=10, type=int)

    parser.add_argument("--scoreThr", default=0.95, type=float, help='fixmatch pseudo-label threshold')
    parser.add_argument("--fixWeight", default=1.0, type=float, help='coefficient of fixMatch loss')
    parser.add_argument('--useInterleave', default="True", help='use EMA model in training')

    # misc
    parser.add_argument("--debug", default="False")
    # endregion
    args = setArgs(parser.parse_args(), params)
    return args


if __name__ == "__main__":
    exec()
