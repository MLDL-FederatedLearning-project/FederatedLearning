import csv
import pandas as pd
import numpy as np
from collections import OrderedDict, defaultdict
from options import args_parser
import os
args = args_parser()
dir_path = os.path.dirname(os.path.realpath(file))
with open(dir_path+"federated_train_alpha_"+args.alpha+".csv") as filecsv:
    lettore = csv.reader(filecsv,delimiter=",")


header_name=[0,1,2]

def get_users_groups_alpha_balanced(args):
    dict_from_csv= pd.read_csv(args.data_dir + "/cifar10_alpha/federated_train_alpha_"+args.alpha+".csv", header=0)
    new=dict_from_csv.groupby('user_id')['image_id'].apply(set).to_dict()
    #print(new[1])
    labels= dict_from_csv['class']
    return labels, new
