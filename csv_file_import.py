import csv
import pandas as pd
import numpy as np
from collections import OrderedDict, defaultdict
from options import args_parser
import os
args = args_parser()
dir_path = os.path.dirname(os.path.realpath(__file__))
with open(dir_path+"/cifar10_alpha/federated_train_alpha_"+str(args.alpha)+".csv") as filecsv:
    lettore = csv.reader(filecsv,delimiter=",")


header_name=[0,1,2]

def get_users_groups_alpha_balanced(args):
    dict_from_csv= pd.read_csv(dir_path + "/cifar10_alpha/federated_train_alpha_"+str(args.alpha)+".csv", header=0)
    new=dict_from_csv.groupby('user_id')['image_id'].apply(set).to_dict()
    print(new)
    labels= dict_from_csv['class']
    return labels, new
