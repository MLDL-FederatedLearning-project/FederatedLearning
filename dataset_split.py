import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import pandas as pd
import random
import collections
from collections import defaultdict
from sampling import get_server,get_dict_labels,random_number_images, non_iid_unbalanced, iid_unbalanced, non_iid_balanced, iid_balanced
import argparse
from options import args_parser

#from options_FedMA import add_fit_args

args = args_parser()
def get_train_valid_loader(args,
                           valid_size=0.2,
                           shuffle=True,
                           pin_memory=False,
                           ):
    """
    Utility function for loading and returning train and valid
    multi-process iterators over the CIFAR-10 dataset.
    If using CUDA, num_workers should be set to 1 and pin_memory to True.
    Params
    ------
    - DATA_DIR: path directory to the dataset.
    - BATCH_SIZE: how many samples per batch to load.
    - augment: whether to apply the data augmentation scheme
      mentioned in the paper. Only applied on the train split.
    - random_seed: fix seed for reproducibility.
    - valid_size: percentage split of the training set used for
      the validation set. Should be a float in the range [0, 1].
    - shuffle: whether to shuffle the train/validation indices.
    - show_sample: plot 9x9 sample grid of the dataset.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
    Returns
    -------
    - train_loader: training set iterator.
    - valid_loader: validation set iterator.
    """
    error_msg = "[!] valid_size should be in the range [0, 1]."
    assert ((valid_size >= 0) and (valid_size <= 1)), error_msg

    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )

    # define transforms
    valid_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
    ])

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    # load the dataset
    train_dataset = datasets.CIFAR10(
        root=args.data_dir, train=True,
        download=True, transform=train_transform,
    )

    valid_dataset = datasets.CIFAR10(
        root=args.data_dir, train=True,
        download=True, transform=valid_transform,
    )

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(args.seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
              train_dataset, batch_size=args.batch_size, sampler=train_sampler,
              pin_memory=pin_memory,drop_last=False)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=args.batch_size, sampler=valid_sampler,
        pin_memory=pin_memory,drop_last=False)



    return (train_loader, valid_loader)


def get_test_loader(args,
                    shuffle=True,
                    pin_memory=False):
    """
    Utility function for loading and returning a multi-process
    test iterator over the CIFAR-10 dataset.
    If using CUDA, num_workers should be set to 1 and pin_memory to True.
    Params
    ------
    - DATA_DIR: path directory to the dataset.
    - BATCH_SIZE: how many samples per batch to load.
    - shuffle: whether to shuffle the dataset after every round.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
    Returns
    -------
    - data_loader: test set iterator.
    """

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    # define transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    dataset = datasets.CIFAR10(
        root=args.data_dir, train=False,
        download=True, transform=transform,
    )
    if args.centralized ==1:
          data_loader = torch.utils.data.DataLoader(
              dataset, batch_size=args.batch_size, shuffle=shuffle)
    else:
            data_loader = torch.utils.data.DataLoader(
              dataset, batch_size=args.batch_size, shuffle=shuffle
         )

    return data_loader


def get_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """
    normalize_train = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )
    normalize_test = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize_train,
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize_test,
    ])
    train_dataset = datasets.CIFAR10(
        root=args.data_dir, train=True,
        download=True, transform=train_transform,
    )
    test_dataset = datasets.CIFAR10(
        root=args.data_dir, train=False,
        download=True, transform=test_transform,
    )
    return train_dataset, test_dataset

def get_user_groups(args):
    train_dataset, _ = get_dataset(args)
    server_data, server_labels, server_id = get_server(train_dataset)

    if args.iid == 0 and args.balanced == 1:
        server_labels, dict_users, traindata_cls_counts = non_iid_balanced(args, server_id, server_labels)

    if args.iid == 0 and args.balanced == 0:
        server_labels, dict_users, traindata_cls_counts= non_iid_unbalanced(args,server_id, server_labels)

    if args.iid == 1 and args.balanced == 1:
        server_labels, new_dict, traindata_cls_counts = iid_balanced(args,server_id, server_labels)

    if args.iid == 1 and args.balanced == 0:
        user_groups = iid_unbalanced(args, server_id, server_labels)

    return server_labels, dict_users, traindata_cls_counts

def get_user_groups_alpha(args):
    train_dataset,_ = get_dataset(args)
    users = np.arange(0,args.num_users)
    server_data, server_labels, server_id = get_server(train_dataset)
    num_items_balanced = int(len(server_id)/args.num_users)
    #print(server_labels)
    min_size=0
    while min_size < 10:
         idx_batch = [[] for _ in range(args.num_users)]
         for k in range(args.num_classes):
            idx_k=[]
            for i in range(len(server_labels)):
                if server_labels[i]==k:
                    idx_k.append(i)

            #print("idx_k", idx_k)
            np.random.shuffle(idx_k)
            #print("idx_k_new", idx_k)
            proportions=np.random.dirichlet(np.repeat(args.alpha, args.num_users))
            #print("proportions1", proportions)
            proportions = np.array([p*(len(idx_j)<len(server_labels)/(args.num_users*args.frac)) for p,idx_j in zip(proportions,idx_batch)])
            #print("proportions2", proportions)
            proportions=proportions/proportions.sum()
            #print("proportions3", proportions)
            proportions = (np.cumsum(proportions)*len(idx_k)).astype(int)[:-1]
            #print("proportions4", proportions)
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])
    dict_users={}
    for j in range(args.num_users):
        np.random.shuffle(idx_batch[j])
        dict_users[j] = idx_batch[j]
        #print(len(dict_users[j]), len(idx_batch[j]))

    return dict_users
'''
args=args_parser()
train_dataset, _= get_dataset(args)
server_data, server_labels, server_id= get_server(train_dataset)
new_dict, nets_cls_counts= iid_unbalanced(args,server_id, server_labels)

print(nets_cls_counts)
'''
