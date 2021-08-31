import torch
import torchvision
from torchvision import datasets, transforms
import numpy as np
import pandas as pd
import collections
import random
from collections import defaultdict
import argparse
#from options_FedMA import add_fit_args



def get_server(train_dataset):
    server_data=[]
    server_id=[]
    server_labels=[]
    for i in range(len(train_dataset)):
        server_data.append(train_dataset[i][0])
        server_labels.append(train_dataset[i][1])
        server_id.append(i)
    return server_data, server_labels, server_id

# We initialize the dict_labels
def get_dict_labels (args,server_id,server_labels):
    dict_labels={}
    num_classes = args.num_classes
    labels = np.arange(0, num_classes)  # the 10 classes we have : from 0 to 9
    for label in labels:
        if label not in dict_labels:
            dict_labels[label] = []
    # We create a dictionary of labels in which we have as keys the labels and the values the indexes of the images
    for i in range(len(server_labels)):
        for label in labels:
            # print(server_labels[i])
            if label == server_labels[i]:
                dict_labels[label].append(server_id[i])
    return dict_labels
    # print(dict_labels)

# This function defines for n clients ( for us n=args.num_users ) how many images to take

def random_number_images(n, args,server_id): #how do we choose n?
    SEED = args.seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    left = len(server_id)
    items = []
    while n > 1:
        average = left / n
        item = int(random.uniform(10, average * 3.1))
        left = left - item
        items.append(item)
        # print(item)
        n = n - 1
    # print(left)
    return np.array(items)


def non_iid_unbalanced(server_id,args):
    #num_users = args.num_users
    users = np.arange(0, args.num_users)
    num_items_unbalanced = random_number_images(args.num_users+1, server_id)  # it respresents the number of imagis each user has for the unbalanced split of the dataset
    dict_users = {}
    all_idxs = [i for i in range(len(server_id))]
    for user in users:
      dict_users[user] = set(np.random.choice(all_idxs, num_items_unbalanced[user], replace=False))
      all_idxs = list(set(all_idxs) - dict_users[user])

    return dict_users

def non_iid_balanced(args,server_id):
    num_users = args.num_users
    users = np.arange(0, num_users)
    num_items_balanced = int(len(server_id)/num_users) # it respresents the number of images each user has for the balanced split of the dataset; each user has the same number of images
    dict_users = {}
    all_idxs = [i for i in range(len(server_id))]
    for user in users:
      dict_users[user] = set(np.random.choice(all_idxs, num_items_balanced, replace=False))
      all_idxs = list(set(all_idxs) - dict_users[user])

    return dict_users

def iid_balanced(args, server_id, server_labels):
    num_users = args.num_users
    users = np.arange(0, num_users)
    num_items_balanced = int(len(server_id) / num_users)
    dict_users = collections.defaultdict(dict)
    labels = np.arange(0, args.num_classes)
    dict_labels = get_dict_labels(args, server_id, server_labels)
    all_idxs = [i for i in range(len(server_id))]
    new_dict = {}
    nets_cls_counts = collections.defaultdict(dict)
    for user in users:
        for label in labels:
            dict_users[user][label] = set(
                np.random.choice(dict_labels[label], int(num_items_balanced / args.num_classes), replace=False))
            all_idxs = list(set(all_idxs) - dict_users[user][label])
            nets_cls_counts[user][label] = len(list(dict_users[user][label]))
        new_dict[user] = set().union(dict_users[user][0], dict_users[user][1], dict_users[user][2],
                                     dict_users[user][3], dict_users[user][4], dict_users[user][5],
                                     dict_users[user][6], dict_users[user][7], dict_users[user][8],
                                     dict_users[user][9])

    return new_dict, nets_cls_counts



def iid_unbalanced(args,server_id, server_labels):
    num_users = args.num_users
    users = np.arange(0, num_users)
    num_items_unbalanced = random_number_images(num_users+1, args,server_id)
    dict_users = collections.defaultdict(dict)
    labels = np.arange(0, args.num_classes)
    dict_labels = get_dict_labels(args, server_id, server_labels)
    all_idxs = [i for i in range(len(server_id))]
    new_dict={}
    for user in users:
        for label in labels:
          dict_users[user][label] = set(np.random.choice(dict_labels[label], int(num_items_unbalanced[user] / args.num_classes), replace=False))
          all_idxs = list(set(all_idxs) - dict_users[user][label])
        new_dict[user] = set().union(dict_users[user][0], dict_users[user][1], dict_users[user][2], dict_users[user][3],
                                     dict_users[user][4], dict_users[user][5], dict_users[user][6], dict_users[user][7],
                                     dict_users[user][8], dict_users[user][9])
    return new_dict

'''
if __name__ == '__main__':
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        train_dataset = datasets.CIFAR10(
            root=data_dir, train=True,
            download=True, transform=train_transform,
        )
'''

