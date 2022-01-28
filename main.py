#!/usr/bin/env python
# coding: utf-8

from __future__ import absolute_import, division, print_function

import random
import os
import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.activation import LeakyReLU
from torch.nn.modules.loss import BCEWithLogitsLoss
from torchvision import transforms

from utils.dataset import GraphDataset
from utils.lr_scheduler import LR_Scheduler
from tensorboardX import SummaryWriter
from helper import Trainer, Evaluator, collate
from option import Options

# from utils.saliency_maps import *

from models.GraphTransformer import Classifier
from models.weight_init import weight_init

from datetime import datetime
from draw import get_loss_curve, get_accuracy_curve
from pytorchtools import EarlyStopping

def seed_everything(seed):
    """
    Seeds basic parameters for reproductibility of results
    
    Arguments:
        seed {int} -- Number of the seed
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    # for reproductibility
    seed_everything(1001)

    args = Options().parse()
    n_class = args.n_class

    torch.cuda.synchronize()
    torch.backends.cudnn.deterministic = True

    data_path = args.data_path
    model_path = args.model_path
    if not os.path.isdir(model_path.split("/")[0]): os.mkdir(model_path.split("/")[0])
    if not os.path.isdir(model_path): os.mkdir(model_path)
    log_path = args.log_path
    if not os.path.isdir(log_path): os.mkdir(log_path)
    # task name for naming saved model files and log files
    task_name = args.task_name

    print(task_name)
    ###################################
    # default false for train, test, graphcam
    train = args.train
    test = args.test
    graphcam = args.graphcam
    print("train:", train, "test:", test, "graphcam:", graphcam)

    ##### Load datasets
    print("preparing datasets and dataloaders......")
    # 8 for training validation and 1 for testing
    batch_size = args.batch_size

    # training
    if train:
        ids_train = open(args.train_set).readlines()
        # print(ids_train)
        # return sample dict contains label, id(name), features, adj
        dataset_train = GraphDataset(os.path.join(data_path, ""), ids_train)
        dataloader_train = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=batch_size, num_workers=8, collate_fn=collate, shuffle=True, pin_memory=True, drop_last=True)
        # batch size: 8
        total_train_num = len(dataloader_train) * batch_size

    # validation or testing
    ids_val = open(args.val_set).readlines()
    dataset_val = GraphDataset(os.path.join(data_path, ""), ids_val)
    dataloader_val = torch.utils.data.DataLoader(dataset=dataset_val, batch_size=batch_size, num_workers=8, collate_fn=collate, shuffle=False, pin_memory=True)
    total_val_num = len(dataloader_val) * batch_size
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ##### creating models #############
    print("creating models......")

    # args.num_epochs = 120
    num_epochs = args.num_epochs
    # args.lr = 1e-3 
    learning_rate = args.lr

    model = Classifier(n_class)
    model = nn.DataParallel(model)

    # for load model (testing and GraphCAM visualization)
    if args.resume:
        print('load model{}'.format(args.resume))
        model.load_state_dict(torch.load(args.resume))

    if torch.cuda.is_available():
        model = model.cuda()
    #model.apply(weight_init)

    #lr: 1e-3, weight decay: 5e-4
    optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate, weight_decay = 4e-6)       # best:5e-4, 4e-3
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20,100], gamma=0.1) # gamma=0.3  # 30,90,130 # 20,90,130 -> 150

    ##################################

    # criterion = nn.CrossEntropyLoss()
    # criterion = BCEWithLogitsLoss()

    if not test:
        # ../graph_transformer/runs/GraphVIT
        writer = SummaryWriter(log_dir=log_path + task_name)
        f_log = open(os.path.join(log_path, task_name + ".log"), 'w')

    trainer = Trainer(n_class)
    evaluator = Evaluator(n_class)

    best_pred = 0.0

    start_time = datetime.now()
    print("Start Time: {start_time}")

    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []

    early_stopping = EarlyStopping(verbose=True)

    # num_epochs 120 for training validation and 1 for testing
    for epoch in range(num_epochs):
        # optimizer.zero_grad()
        model.train()
        train_loss = 0.
        val_loss = 0
        total = 0.

        current_lr = optimizer.param_groups[0]['lr']
        print('\n=>Epoches %i, learning rate = %.7f, previous best = %.4f' % (epoch+1, current_lr, best_pred))

        if train:
            for i_batch, sample_batched in enumerate(dataloader_train):
                #scheduler(optimizer, i_batch, epoch, best_pred)

                # 1 batch
                preds,labels,loss = trainer.train(sample_batched,  model)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # scheduler.step()

                train_loss += loss
                total += len(labels)

                trainer.metrics.update(labels, preds)
                #trainer.plot_cm()

                # log_interval_local:6 (print every batch size x log_interval_local: 8 x 6 = 48)
                if (i_batch + 1) % args.log_interval_local == 0:
                    print("[%d/%d] train loss: %.3f; train acc: %.3f" % (total, total_train_num, train_loss / total, trainer.get_scores()))
                    trainer.plot_cm()
        
        # print the last one (total) [208/208]
        if not test: 
            print("[%d/%d] train loss: %.3f; train acc: %.3f" % (total_train_num, total_train_num, train_loss / total, trainer.get_scores()))
            train_losses.append((train_loss / total).item())
            train_accs.append(trainer.get_scores())
            trainer.plot_cm()


        # applies to every epoch (validation) and testing one epoch
        if epoch % 1 == 0:
            with torch.no_grad():
                model.eval()
                print("evaluating...")

                total = 0.
                batch_idx = 0

                # batch size: 
                for i_batch, sample_batched in enumerate(dataloader_val):
                    #pred, label, _ = evaluator.eval_test(sample_batched, model)
                    preds, labels, loss = evaluator.eval_test(sample_batched, model, graphcam)

                    total += len(labels)
                    val_loss += loss

                    evaluator.metrics.update(labels, preds)

                    # log_interval_local:6 (print every batch size x log_interval_local: 8 x 6 = 48)
                    if (i_batch + 1) % args.log_interval_local == 0:
                        print('[%d/%d] val loss: %.3f; val acc: %.3f' % (total, total_val_num, val_loss / total, evaluator.get_scores()))
                        evaluator.plot_cm()

                # print the last one [208/208]
                print('[%d/%d] val loss: %.3f; val acc: %.3f' % (total_val_num, total_val_num, val_loss / total, evaluator.get_scores()))
                val_losses.append((val_loss / total).item())
                val_accs.append(evaluator.get_scores())
                evaluator.plot_cm()

                # torch.cuda.empty_cache()

                val_acc = evaluator.get_scores()
                if val_acc > best_pred: 
                    best_pred = val_acc
                    if not test:
                        print("saving model...")
                        # ../graph_transformer/saved_models/GraphVIT_{epoch}.pth
                        torch.save(model.state_dict(), os.path.join(model_path, task_name + ".pth"))

                log = ""
                log = log + 'epoch [{}/{}] ------ train acc = {:.4f}, val acc = {:.4f}'.format(epoch+1, num_epochs, trainer.get_scores(), evaluator.get_scores()) + "\n"

                log += "================================\n"
                print(log)
                if test:
                    break

                f_log.write(log)
                f_log.flush()

                writer.add_scalars('accuracy', {'train acc': trainer.get_scores(), 'val acc': evaluator.get_scores()}, epoch+1)

                # early stopping
                early_stopping((val_loss / total).item(), model)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break

        trainer.reset_metrics()
        evaluator.reset_metrics()

    if not test: f_log.close()

    print(f"Training Execution time: {datetime.now() - start_time}")

    if train:
        get_loss_curve(args.figure_path, train_losses, val_losses)
        get_accuracy_curve(args.figure_path, train_accs, val_accs)

if __name__ == "__main__":
    main()