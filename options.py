import argparse
import os

dir_path = os.path.dirname(os.path.realpath(__file__))

def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default=dir_path,
                        help="directory of the project")
    parser.add_argument('--epochs', type=int, default=15,
                        help="number of global rounds of training")
    parser.add_argument('--communication_rounds', type=int, default=500,
                        help="number of communication rounds")
    parser.add_argument('--batch_size', type=int, default=64,
                        help="BATCH_SIZE")
    parser.add_argument('--num_users', type=int, default=100,
                        help="number of users: K")
    parser.add_argument('--frac', type=float, default=0.1,
                        help='the fraction of clients: C')
    parser.add_argument('--local_ep', type=int, default=2,
                        help="the number of local epochs: E")
    parser.add_argument('--local_batch_size', type=int, default=64,
                        help="local batch size: B")
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum')
    parser.add_argument('--weight_decay', type=float, default=0.0004,
                        help='weight_decay')
    parser.add_argument('--model', type=str, default='cnn', help='model name')
    parser.add_argument('--alpha', type=float, default=0.1,
                        help="for personalization")
    parser.add_argument('--alpha_use', type=str, default='True',
                        help="use of alpha for partition")
    parser.add_argument('--dataset', type=str, default='cifar10', help="name \
                        of dataset")
    parser.add_argument('--num_classes', type=int, default=10, help="number \
                        of classes")
    parser.add_argument('--gpu', default=None, help="To use cuda, set \
                        to a specific GPU ID. Default set to use CPU.")
    parser.add_argument('--optimizer', type=str, default='sgd', help="type \
                        of optimizer")
    parser.add_argument('--iid', type=int, default=1,
                        help='Default set to iid=1. For non iid use 0')
    parser.add_argument('--balanced', type=int, default=1,
                        help='Default set to balanced =1. For unbalanced set to 0')
    parser.add_argument('--verbose', type=int, default=1, help='verbose')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--n_nets', type=int, default=10, metavar='NN',
                        help='number of workers in a distributed cluster')
    parser.add_argument('--comm_type', type=str, default='fedavg',
                        help='which type of communication strategy is going to be used: fedavg/pfnm')
    parser.add_argument('--net_config', type=list, default=[3072, 100, 10])

    args = parser.parse_args()
    return args
