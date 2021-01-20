import torch
import torchvision
import torchvision.models.resnet as torchvision_resnet
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import random
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


# it makes dgx1 available for plot.
def get_args():
    parser = argparse.ArgumentParser(description='A general training enviroment for different learning methods.')


    parser.add_argument('--cuda',  action='store_true', help='use cuda')

    # For output 
    parser.add_argument('--name', default='result', type=str,
                        help='the path to save the result')

    # For version control
    parser.add_argument('--version', default='0', type=str,
                        help='version of the same params.')

    # For methods
    parser.add_argument('--method', '-r', metavar='The learning method.', default='sgd',)
    parser.add_argument('-b', '--batch-size', default=128, type=int,
                        metavar='N', help='mini-batch size (default: 128)')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                        metavar='W', help='weight decay (e.g. 5e-4)')
    parser.add_argument('--decrease-freq', default=50, type=int, metavar='N',
                        help='epochs that decrease lr(default: 50)')
    parser.add_argument('--epochs', type=int, help='epoch number', default=30)

    # For Data
    parser.add_argument('--data-path', default='~/Data', type=str,help='The path of datasets')
    parser.add_argument('--data', type=str, help='points or mnist', default='cifar10')
    
    # For models
    parser.add_argument('--arch', metavar='ARCH', default='mgnet')

    return parser.parse_args()


def main():
    args = get_args()  # get the arguments

    args.name = 'm={},net={},ds={},ep={},lr={},decrease_freq={},wd={},momentum={},batch_size={},v={}'
    args.name = args.name.format(args.method,
                                 args.arch,
                                 args.data,
                                 args.epochs,
                                 args.lr,
                                 args.decrease_freq,
                                 args.weight_decay,
                                 args.momentum,
                                 args.batch_size,
                                 args.version
                                 )
    # file name
    training_file_name = args.name + '.train'
    validation_file_name = args.name + '.validate'
    lr_file_name = args.name + '.lr'

    # example of files for training  
    f = open(training_file_name, 'a')

if __name__ == '__main__':
    main()
