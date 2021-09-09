import argparse
import os

dir_path = os.path.dirname(os.path.realpath(__file__))

def args_parser():
    parser = argparse.ArgumentParser()

    # federated arguments (Notation for the arguments followed from paper)
    parser.add_argument('--data_dir', type=str, default=dir_path,
                        help="directory of the project")
    parser.add_argument('--centralized', type=int, default=1,
                        help="default value = 1 for the centralized model, for the federated put 0")
    parser.add_argument('--epochs', type=int, default=50,
                        help="number of rounds of training")
    parser.add_argument('--communication_rounds', type=int, default=50,
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

    # model arguments
    parser.add_argument('--model', type=str, default='cnn', help='model name')
    parser.add_argument('--kernel_num', type=int, default=9,
                        help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to \
                        use for convolution')
    parser.add_argument('--num_channels', type=int, default=1, help="number \
                        of channels of imgs")
    parser.add_argument('--norm', type=str, default='batch_norm',
                        help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32,
                        help="number of filters for conv nets -- 32 for \
                        mini-imagenet, 64 for omiglot.")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than \
                        strided convolutions")
    parser.add_argument('--alpha', type=int, default=0.05,
                        help="for personalization")
    parser.add_argument('--alpha_use', type=str, default='False',
                        help="put true if want to use alpha")

    # other arguments
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
    parser.add_argument('--balanced', type=int, default=0,
                        help='Default set to balanced =1. For unbalanced set to 0')
    parser.add_argument('--stopping_rounds', type=int, default=10,
                        help='rounds of early stopping')
    parser.add_argument('--verbose', type=int, default=1, help='verbose')
    parser.add_argument('--seed', type=int, default=1, help='random seed')


    #experiment arguments
    parser.add_argument('--partition', type=str, default='homo', metavar='N',
                        help='how to partition the dataset on local workers')
    parser.add_argument('--retrain_lr', type=float, default=0.01, metavar='RLR',
                        help='learning rate using in specific for local network retrain (default: 0.01)')
    parser.add_argument('--retrain_epochs', type=int, default=1, metavar='REP',
                        help='how many epochs will be trained in during the locally retraining process')
    parser.add_argument('--partition_step_size', type=int, default=6, metavar='PSS',
                        help='how many groups of partitions we will have')
    parser.add_argument('--local_points', type=int, default=5000, metavar='LP',
                        help='the approximate fixed number of data points we will have on each local worker')
    parser.add_argument('--partition_step', type=int, default=0, metavar='PS',
                        help='how many sub groups we are going to use for a particular training process')
    parser.add_argument('--n_nets', type=int, default=10, metavar='NN',
                        help='number of workers in a distributed cluster which should be multiply of number of users '
                             'and fraction of the user')
    parser.add_argument('--retrain', type=bool, default=False,
                        help='whether to retrain the model or load model locally')
    parser.add_argument('--minor-frac', type=float, default=0.05, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--minor-frac-test', type=float, default=0.5, metavar='N',
                        help='the fraction of gray scale images in the test set')
    parser.add_argument('--rematching', type=bool, default=False,
                        help='whether to recalculating the matching process (this is for speeding up the debugging process)')
    parser.add_argument('--oneshot_matching', type=bool, default=False, metavar='OM',
                        help='if the code is going to conduct one shot matching')
    parser.add_argument('--comm_type', type=str, default='fedavg',
                        help='which type of communication strategy is going to be used: fedma/fedavg/...layerwise/blockwise')
    parser.add_argument('--comm_round', type=int, default=2,
                        help='how many round of communications we shoud use')
    parser.add_argument('--net_config', type=list, default=[3072, 100, 10])
    args = parser.parse_args()
    return args
