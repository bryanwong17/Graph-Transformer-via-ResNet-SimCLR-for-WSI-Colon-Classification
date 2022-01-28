###########################################################################
# Created by: YI ZHENG
# Email: yizheng@bu.edu
# Copyright (c) 2020
###########################################################################

import os
import argparse
import torch

class Options():
    def __init__(self):
        parser = argparse.ArgumentParser(description='PyTorch Classification')
        # model and dataset 
        parser.add_argument('--n_class', type=int, default=2, help='classification classes')
        parser.add_argument('--data_path', type=str, default="graphs_test_4/simclr_files_2", help='path to dataset where images store')
        parser.add_argument('--train_set', type=str, default="train_set.txt", help='train')
        parser.add_argument('--val_set', type=str, default="test_set.txt", help='validation')
        parser.add_argument('--model_path', type=str, default="result_test_4/saved_models", help='path to trained model')
        parser.add_argument('--log_path', type=str, default="result_test_4/runs", help='path to log files')
        parser.add_argument('--task_name', type=str, default="GraphVIT", help='task name for naming saved model files and log files')
        parser.add_argument('--train', action='store_true', default=False, help='train only')
        parser.add_argument('--test', action='store_true', default=True, help='test only')
        parser.add_argument('--batch_size', type=int, default=8, help='batch size for origin global image (without downsampling)')
        parser.add_argument('--log_interval_local', type=int, default=6, help='classification classes')
        parser.add_argument('--resume', type=str, default="result_test_4/saved_models/GraphVIT.pth", help='path for model')
        parser.add_argument('--graphcam', action='store_true', default=False, help='GraphCAM')
        parser.add_argument('--figure_path', type=str, default="result_test_4/curve", help='GraphCAM')

        # the parser
        self.parser = parser

    def parse(self):
        args = self.parser.parse_args()
        # default settings for epochs and lr

        args.num_epochs = 120
        args.lr = 4e-5             

        if args.test:
            args.num_epochs = 1
        return args
