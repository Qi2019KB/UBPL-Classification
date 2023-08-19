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
from models.base.initStrategy import InitializeStrategy as InitS

from utils.base.log import Logger
from utils.base.comm import CommUtils as comm
from utils.process import ProcessUtils as proc
from utils.parameters import consWeight_increase, pseudoWeight_increase, FDLWeight_decrease
from utils.losses import ClassLoss, ClassFixLoss, ClassDistLoss, ClassPseudoLoss, AvgCounter
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
    models, models_ema, optims, model_pNum = [], [], [], 0
    for bIdx in range(args.brNum):
        model = modelClass.__dict__["ClassModel"](modelType=args.model, num_classes=args.num_classes, mode=args.feature_mode, dataset=args.dataset).to(args.device)
        if args.diffInit and bIdx == 1: InitS.parameters_initialize(model, "xavier")
        models.append(model)
        model_ema = ModelEMA(model, args.ema_decay, args)
        models_ema.append(model_ema)

        no_decay = ['bias', 'bn']
        grouped_parameters = [{'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
                              {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
        optims.append(TorchSGD(grouped_parameters, lr=args.lr, momentum=args.momentum, nesterov=args.nesterov))
        model_pNum += (sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in model_ema.ema.parameters()))
    logc = "=> initialized MFDSs ({}) Structure (params: {})".format(args.model, format(model_pNum / (1024**2), ".2f"))
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
        args.FDLWeight = FDLWeight_decrease(epo, args)
        args.pseudoWeight = pseudoWeight_increase(epo, args)
        # endregion

        # region 2.2 model training and validating
        startTM = datetime.datetime.now()
        total_losses, clc_losses, fix_losses, mtc_losses, epc_losses, fdc_loss, fix_mask, mtc_mask, epc_mask = train(trainLoader, models, models_ema, optims, args)
        logger.print("L3", "model training finished...", start=startTM)

        startTM = datetime.datetime.now()
        predsArray, accs = validate(validLoader, models_ema, args)
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
                      "model": args.model, "feature_mode": args.feature_mode}
        for bIdx in range(args.brNum):
            checkpoint["model{}_state".format(bIdx+1)] = models[bIdx].state_dict()
            checkpoint["model{}_ema_state".format(bIdx+1)] = (models_ema[bIdx].ema.module if hasattr(models_ema[bIdx].ema, "module") else models_ema[bIdx].ema).state_dict()
            checkpoint["optim{}_state".format(bIdx+1)] = optims[bIdx].state_dict()
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
        log_data = {"total_losses": total_losses, "clc_losses": clc_losses, "fix_losses": fix_losses,
                    "mtc_losses": mtc_losses, "epc_losses": epc_losses, "fdc_loss": fdc_loss,
                    "fix_mask": fix_mask, "mtc_mask": mtc_mask, "epc_mask": epc_mask,
                    "AP": accs[0], "mAP": accs[1]}
        comm.json_save(log_data, "{}/logs/logData/logData_{}.json".format(args.basePath, epo+1), isCover=True)

        # Pseudo-labels data storage
        pseudo_data = {"predsArray": predsArray}
        comm.json_save(pseudo_data, "{}/logs/pseudoData/pseudoData_{}.json".format(args.basePath, epo+1), isCover=True)
        logger.print("L3", "log storage finished...", start=startTM)
        # endregion

        # region 2.5 output result
        # Training performance
        marks = ["mds1", "mds2"]
        for idx in range(len(models)):
            fmtc = "[{}/{} | {} | lr: {} | fixW: {}, consW: {}, pseudoW: {}, FDLW: {} | fix_mask: {}, mtc_mask: {}, epc_mask: {}] total_loss: {}, clc_loss: {}, fix_loss: {}, mtc_loss: {}, epc_loss: {}, fdc_loss: {}"
            logc = fmtc.format(format(epo + 1, "3d"), format(args.epochs, "3d"), marks[idx],
                               format(optims[0].state_dict()['param_groups'][0]['lr'], ".3f"),
                               format(args.fixWeight, ".2f"), format(args.consWeight, ".2f"), format(args.pseudoWeight, ".2f"), format(args.FDLWeight, ".2f"),
                               format(fix_mask[idx], ".3f"), format(mtc_mask[idx], ".3f"), format(epc_mask[idx], ".3f"),
                               format(total_losses[idx], ".5f"), format(clc_losses[idx], ".5f"), format(fix_losses[idx], ".5f"),
                               format(mtc_losses[idx], ".5f"), format(epc_losses[idx], ".5f"), format(fdc_loss, ".10f"))
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


def train(trainLoader, models, models_ema, optims, args):
    # region 1. Preparation
    total_counters = [AvgCounter() for _ in range(len(models))]
    clc_counters = [AvgCounter() for _ in range(len(models))]
    fix_counters = [AvgCounter() for _ in range(len(models))]
    mtc_counters = [AvgCounter() for _ in range(len(models))]
    epc_counters = [AvgCounter() for _ in range(len(models))]
    fdc_counter = AvgCounter()
    fix_mask_counters = [AvgCounter() for _ in range(len(models))]
    mtc_mask_counters = [AvgCounter() for _ in range(len(models))]
    epc_mask_counters = [AvgCounter() for _ in range(len(models))]
    class_criterion = ClassLoss().to(args.device)
    fix_criterion = ClassFixLoss(scoreThr= args.scoreThr).to(args.device)
    consistency_criterion = ClassDistLoss(scoreThr= args.scoreThr).to(args.device)
    pseudo_criterion = ClassPseudoLoss(is_weight_mean=True).to(args.device)
    for model in models: model.train()
    for model_ema in models_ema: model_ema.ema.train()
    # endregion

    # region 2. Training
    for bat, (imgs_labeled, imgs_strong, imgs_weak, labels, meta) in enumerate(trainLoader):
        for optim in optims: optim.zero_grad()

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
        outs_labeled_f = []
        for mIdx in range(len(models)):
            outs_labeled_m, outs_weak_m, outs_strong_m = [], [], []
            outs_labeled_f_m = []
            for aIdx in range(len(imgs_labeled)):
                bs = imgs_labeled[aIdx].shape[0]
                if args.useInterleave:
                    inputs_cat = interleave(torch.cat((imgs_labeled[aIdx], imgs_weak[aIdx], imgs_strong[aIdx])), 2*args.mu+1).to(args.device)
                else:
                    inputs_cat = torch.cat((imgs_labeled[aIdx], imgs_weak[aIdx], imgs_strong[aIdx])).to(args.device)
                outs_cat, features_cat = models[mIdx](inputs_cat)  # outs: [bs, k]; labels: [bs]
                if args.useInterleave:
                    outs_cat = de_interleave(outs_cat, 2 * args.mu + 1)
                    features_cat = de_interleave(features_cat, 2 * args.mu + 1)
                outs_labeled_m.append(outs_cat[:bs])
                out_weak, out_strong = outs_cat[bs:].chunk(2)
                outs_weak_m.append(out_weak)
                outs_strong_m.append(out_strong)

                outs_labeled_f_m.append(features_cat[:bs])

            outs_labeled.append(torch.stack(outs_labeled_m, dim=0))
            outs_weak.append(torch.stack(outs_weak_m, dim=0))
            outs_strong.append(torch.stack(outs_strong_m, dim=0))

            outs_labeled_f.append(torch.stack(outs_labeled_f_m, dim=0))

        outs_labeled = torch.stack(outs_labeled, dim=0)
        outs_weak = torch.stack(outs_weak, dim=0)
        outs_strong = torch.stack(outs_strong, dim=0)

        outs_labeled_f = torch.stack(outs_labeled_f, dim=0)

        outs_weak_ema = []
        with torch.no_grad():
            for mIdx in range(len(models_ema)):
                outs_weak_ema_m = []
                for aIdx in range(len(imgs_labeled)):
                        out_weak, _ = models_ema[mIdx].ema(imgs_weak[aIdx])
                        outs_weak_ema_m.append(out_weak)
                outs_weak_ema.append(torch.stack(outs_weak_ema_m, dim=0))
            outs_weak_ema = torch.stack(outs_weak_ema, dim=0)
        # endregion

        # region 2.3 classification constraint
        clc_losses = [0. for _ in range(len(models))]
        for mIdx in range(len(models)):
            clc_sum, clc_count = 0., 0
            for aIdx in range(len(outs_labeled[mIdx])):  # [mNum, augNum, bs, k]
                loss, n = class_criterion(outs_labeled[mIdx, aIdx], labels)
                clc_sum += loss
                clc_count += n
            clc_losses[mIdx] = args.classWeight * ((clc_sum / clc_count) if clc_count > 0 else clc_sum)
            clc_counters[mIdx].update(clc_losses[mIdx].item(), clc_count)
        # endregion

        # region 2.4 fix_sum, fix_count
        fix_losses = [0. for _ in range(len(models))]
        for mIdx in range(len(models)):
            fix_sum, fix_count = 0., 0
            mask_sum, mask_num, mask_lens = 0., 0, 0
            for aIdx in range(len(outs_weak[mIdx])):  # [augNum, bs, k]
                loss, n, mask_value, mask_len = fix_criterion(outs_strong[mIdx, aIdx], outs_weak[mIdx, aIdx].detach())
                fix_sum += loss
                fix_count += n
                mask_sum += mask_value.item()
                mask_lens += mask_len
                mask_num += 1
            fix_losses[mIdx] = args.fixWeight * ((fix_sum / fix_count) if fix_count > 0 else fix_sum)
            fix_counters[mIdx].update(fix_losses[mIdx].item(), fix_count)
            fix_mask_counters[mIdx].update(mask_sum/mask_num, mask_num)
        # endregion

        # region 2.5 mean-teacher consistency constraint
        if args.useConsistency:
            mtc_losses = [0. for _ in range(len(models_ema))]
            for mIdx in range(len(models_ema)):
                mtc_sum, mtc_count = 0., 0
                mask_sum, mask_num, mask_lens = 0., 0, 0
                for aIdx in range(len(outs_weak_ema[mIdx])):  # [augNum, bs, k]
                    loss, n, mask_value, mask_len = consistency_criterion(outs_strong[mIdx, aIdx], outs_weak_ema[mIdx, aIdx].detach())
                    mtc_sum += loss
                    mtc_count += n
                    mask_sum += mask_value.item()
                    mask_lens += mask_len
                    mask_num += 1
                mtc_losses[mIdx] = args.consWeight * ((mtc_sum / mtc_count) if mtc_count > 0 else mtc_sum)
                mtc_counters[mIdx].update(mtc_losses[mIdx].item(), mtc_count)
                mtc_mask_counters[mIdx].update(mask_sum/mask_num, mask_num)
        else:
            mtc_losses = [0. for _ in range(len(models))]
            for mtc_counter in mtc_counters: mtc_counter.update(0., 1)
            for mtc_mask_counter in mtc_mask_counters: mtc_mask_counter.update(0., 1)
        # endregion

        # region 2.6 ensenble prediction constraint
        if args.useEnsemblePseudo:
            epc_losses = [0. for _ in range(len(models))]
            for mIdx in range(len(models)):
                epc_sum, epc_count = 0., 0
                mask_sum, mask_num, mask_lens = 0., 0, 0
                for aIdx in range(len(outs_weak_ema[mIdx])):  # [modelNum, augNum, bs, k]
                    loss, n, mask_value, mask_len = pseudo_criterion(outs_strong[mIdx, aIdx], outs_weak_ema[:, aIdx].clone().detach().squeeze())
                    epc_sum += loss
                    epc_count += n
                    mask_sum += mask_value.item()
                    mask_lens += mask_len
                    mask_num += 1
                epc_losses[mIdx] = args.pseudoWeight * ((epc_sum / epc_count) if epc_count > 0 else epc_sum)
                epc_counters[mIdx].update(epc_losses[mIdx].item(), epc_count)
                epc_mask_counters[mIdx].update(mask_sum/mask_num, mask_num)
        else:
            epc_losses = [0. for _ in range(len(models))]
            for epc_counter in epc_counters: epc_counter.update(0., 1)
        # endregion

        # region 2.7. multi-view features decorrelation loss
        if args.useFDL and args.FDLWeight > 0:
            fdc_sum, fdc_count = 0., 0
            for outs_f in [outs_labeled_f]:  # , outs_strong_f, outs_weak_f
                for aIdx in range(outs_f.shape[1]):
                    c_cov, c_num = proc.features_cov(outs_f[0, aIdx].unsqueeze(1), outs_f[1, aIdx].unsqueeze(1))
                    fdc_sum += c_cov
                    fdc_count += c_num
            fdc_loss = args.FDLWeight * ((fdc_sum / fdc_count) if fdc_count > 0 else fdc_sum) / 2
            fdc_counter.update(fdc_loss.item(), fdc_count)
        else:
            fdc_loss = 0.
            fdc_counter.update(0., 1)
        # endregion

        # region 2.8 calculate total loss & update model
        for mIdx in range(len(models)):
            total_loss = clc_losses[mIdx] + mtc_losses[mIdx] + fix_losses[mIdx] + epc_losses[mIdx] + fdc_loss
            total_counters[mIdx].update(total_loss.item(), 1)
            total_loss.backward(retain_graph=True)
        for optim in optims: optim.step()
        for mIdx, model_ema in enumerate(models_ema): model_ema.update(models[mIdx])
        # endregion

        # region 2.9 clearing the GPU Cache
        del outs_labeled, outs_weak, outs_strong, outs_labeled_f, outs_weak_ema
        # endregion
    # endregion

    # region 3. records neaten
    total_records = [counter.avg for counter in total_counters]
    clc_records = [counter.avg for counter in clc_counters]
    fix_records = [counter.avg for counter in fix_counters]
    mtc_records = [counter.avg for counter in mtc_counters]
    epc_records = [counter.avg for counter in epc_counters]
    fdc_record = fdc_counter.avg
    fix_mask_records = [counter.avg for counter in fix_mask_counters]
    mtc_mask_records = [counter.avg for counter in mtc_mask_counters]
    epc_mask_records = [counter.avg for counter in epc_mask_counters]
    # endregion
    return total_records, clc_records, fix_records, mtc_records, epc_records, fdc_record, fix_mask_records, mtc_mask_records, epc_mask_records


def validate(validLoader, models_ema, args):
    # region 1. Preparation
    predsArray, labelsArray = [], []
    for model_ema in models_ema: model_ema.ema.eval()
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
            outs, outs_ema = [], []
            for mIdx in range(len(models_ema)):
                outs_ema.append(models_ema[mIdx].ema(imgs)[0])
            outs_ema = torch.stack(outs_ema, dim=0)  # [modelNum, bs, k]

            # score based prediction
            scores_br, preds_br = [], []
            for mIdx in range(len(models_ema)):
                scores, preds = torch.max(outs_ema[mIdx].data, 1)
                scores_br.append(scores)
                preds_br.append(preds)
            score_preds = []
            for idx in range(len(preds_br[0])):
                if scores_br[0][idx] >= scores_br[1][idx]:
                    score_preds.append(preds_br[0][idx].item())
                else:
                    score_preds.append(preds_br[1][idx].item())
            predsArray += score_preds

            # region 2.3 clearing the GPU Cache
            del outs, outs_ema
            # endregion
    # endregion

    # region 3 calculate the accuracy
    validArray = [precision_score(labelsArray, predsArray, average="micro"),  # AP
                  precision_score(labelsArray, predsArray, average="macro")]  # mAP
    # endregion
    return predsArray, validArray


def interleave(x, size):
    s = list(x.shape)
    return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def de_interleave(x, size):
    s = list(x.shape)
    return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


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
    parser = argparse.ArgumentParser(description="MFDSs Training")

    # Model setting
    parser.add_argument("--model", default="WideResNet", choices=["WideResNet"])
    parser.add_argument("--feature_mode", default="AvgPool", choices=["AvgPool"])
    parser.add_argument("--brNum", default=2, type=int)
    parser.add_argument("--augNum", default=1, type=int)
    parser.add_argument("--diffInit", default="True")

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

    parser.add_argument('--useConsistency', default="True")
    parser.add_argument("--consWeight_max", default=1.0, type=float)
    parser.add_argument("--consWeight_min", default=0.0, type=float)
    parser.add_argument("--consWeight_start", default=5, type=float)
    parser.add_argument("--consWeight_rampup", default=10, type=int)

    parser.add_argument("--useFDL", default="False")
    parser.add_argument("--FDLWeight_max", default=1.0, type=float)
    parser.add_argument("--FDLWeight_min", default=0.0, type=float)
    parser.add_argument("--FDLWeight_rampup", default=20, type=int)

    parser.add_argument("--useEnsemblePseudo", default="True")
    parser.add_argument("--pseudoWeight_max", default=1.0, type=float)
    parser.add_argument("--pseudoWeight_min", default=0.0, type=float)
    parser.add_argument("--pseudoWeight_start", default=5, type=float)
    parser.add_argument("--pseudoWeight_rampup", default=10, type=int)

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
