import copy
import torch
from torchvision import datasets, transforms
#from sampling import random_number_images, non_iid_unbalanced, iid_unbalanced, non_iid_balanced, iid_unbalanced




def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.communication_rounds}\n')

    print('    Federated parameters:')
    if args.iid == 0 and args.balanced == 0:
        print(' NON IID UNBALANCED')
    elif args.iid == 0 and args.balanced == 1:
        print(' NON IID BALANCED')
    elif args.iid == 1 and args.balanced == 1:
            print(' IID BALANCED')
    else:
        print(' IID UNBALANCED')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_batch_size}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return

