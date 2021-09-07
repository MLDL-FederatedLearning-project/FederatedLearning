#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import logging
import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm
import random

import torch
from tensorboardX import SummaryWriter

from options import args_parser
from update import LocalUpdate, test_inference
from models import CNN
from utils import  average_weights, exp_details
from dataset_split import get_dataset, get_user_groups
from models_fedma import pdm_prepare_weights,pdm_prepare_freq,partition_data,compute_pdm_net_accuracy
from models_fedma import pdm_multilayer_group_descent,compute_iterative_pdm_matching,load_new_state,train_net
from itertools import product
from dataset_split import get_train_valid_loader, get_test_loader,get_user_groups_alpha
from update import DatasetSplit

from sampling import random_number_images, non_iid_unbalanced, iid_unbalanced, non_iid_balanced, iid_unbalanced,get_server,iid_balanced



if __name__ == '__main__':
    start_time = time.time()


    # define paths
    path_project = os.path.abspath('../../Downloads')
    logger = SummaryWriter('../logs')

    args = args_parser()
    exp_details(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    if args.gpu:
        torch.cuda.set_device(args.gpu)
    device = 'cuda' if args.gpu else 'cpu'


    # load dataset and user groups
    train_dataset, test_dataset = get_dataset(args)
    _,user_groups,cls_count = get_user_groups(args)
    #user_groups=get_user_groups_alpha(args)
    #print("user_groups",user_groups)


    # BUILD MODEL
    args.model = 'cnn'
    # Convolutional neural network

    args.dataset = 'cifar'
    global_model = CNN(args=args)

    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()
    #print(global_model)

    # copy weights
    global_weights = global_model.state_dict()

    # Training
    train_loss, train_accuracy = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 1
    val_loss_pre, counter = 0, 0



    if args.comm_type == "fedavg":
        for round in range(args.communication_rounds):
            local_weights, local_losses = [], []
            print(f'\n | Global Training Round : {round + 1} |\n')

            global_model.train()
            m = max(int(args.frac * args.num_users), 1)
            idxs_users = np.random.choice(range(args.num_users), m, replace=False)
            for idx in idxs_users:
                local_model = LocalUpdate(args=args, dataset=train_dataset,
                                  idxs=user_groups[idx], logger=logger)

                w, loss = local_model.update_weights(model=copy.deepcopy(global_model), global_round=round)

                local_weights.append(copy.deepcopy(w))
                local_losses.append(copy.deepcopy(loss))
            global_weights = average_weights(local_weights)
            # update global weights
            global_model.load_state_dict(global_weights)

            loss_avg = sum(local_losses) / len(local_losses)
            train_loss.append(loss_avg)

            # Calculate avg training accuracy over all users at every round
            list_acc, list_loss = [], []
            global_model.eval()
            for c in range(args.num_users):
                local_model = LocalUpdate(args=args, dataset=train_dataset,
                                          idxs=user_groups[c], logger=logger)
                acc, loss = local_model.inference(model=global_model)
                # print("Acc:", acc)
                list_acc.append(acc)
                list_loss.append(loss)

            train_accuracy.append(sum(list_acc) / len(list_acc))

            # print global training loss after every 'i' rounds
            # if (round+1) % print_every == 0:
            print(f' \nAvg Training Stats after {round + 1} global rounds:')
            print(f'Training Loss : {np.mean(np.array(train_loss))}')
            print('Train Accuracy: {:.2f}% \n'.format(train_accuracy[-1]))

        # Test inference after completion of training
        test_acc, test_loss = test_inference(args, global_model, test_dataset)

        print(f' \n Results after {args.communication_rounds} global rounds of training:')
        print("|---- Avg Train Accuracy: {:.2f}%".format(train_accuracy[-1]))
        print("|---- Test Accuracy: {:.2f}%".format(test_acc))

        # Saving the objects train_loss and train_accuracy
        dir_path = os.path.dirname(os.path.realpath(__file__))

        file_name = dir_path + '/{}_{}_{}_{}_{}_{}_{}_alpha{}.pkl'. \
            format(args.dataset, args.model, args.communication_rounds, args.num_users, args.frac,
                   args.local_ep, args.local_batch_size, args.alpha)

        with open(file_name, 'wb') as f:
            pickle.dump([train_loss, train_accuracy], f)

        print('\n Total Run Time: {0:0.4f}'.format(time.time() - start_time))

        # PLOTTING (optional)
        import matplotlib
        import matplotlib.pyplot as plt

        matplotlib.use('Agg')

        # Plot Loss curve
        plt.figure()
        plt.title('Training Loss vs Communication rounds')
        plt.plot(range(len(train_loss)), train_loss, color='r')
        plt.ylabel('Training loss')
        plt.xlabel('Communication Rounds')
        plt.savefig(dir_path + '/loss_federated.png'.
                    format(args.dataset, args.model, args.communication_rounds, args.frac,
                           args.iid, args.local_ep, args.local_batch_size))
        # plt.show()

        # # Plot Average Accuracy vs Communication rounds
        plt.figure()
        plt.title('Average Accuracy vs Communication rounds')
        plt.plot(range(len(train_accuracy)), train_accuracy, color='k')
        plt.ylabel('Average Accuracy')
        plt.xlabel('Communication Rounds')
        plt.savefig(dir_path + '/accuracy_federated.png'.
                    format(args.dataset, args.model, args.communication_rounds, args.frac,
                           args.iid, args.local_ep, args.local_batch_size))
        # plt.show()

    elif args.comm_type == "fedma":

        local_weights, local_losses = [], []

        global_model.train()
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        batch_weights = pdm_prepare_weights(global_model)
        n_nets = int(args.num_users * args.frac)
        #print("batch_weights",batch_weights)
        n_classes = args.net_config
        #print("n classes",len(n_classes), type(n_classes))
        n_classes = n_classes[-1]
        cls_freqs=cls_count
        #_,_,_,_,net_dataidx_map,cls_freqs = partition_data(train_dataset, test_dataset, args.n_nets)
        #_,server_labels,server_id=get_server(train_dataset)
        #cls_freqs = non_iid_unbalanced(server_id,args)
        #cls_freqs=iid_balanced(args,server_id,server_labels)
        #print("CLS freq",cls_freqs)
        batch_freqs = pdm_prepare_freq(cls_freqs, n_classes)
        #print("batch frequencies", batch_freqs)
        gammas = [1.0, 1e-3, 50.0]
        sigmas = [1.0, 0.1, 0.5]
        sigma0s = [1.0, 10.0]

        """sigma0s = [1.0]
        sigmas = [1.0]
        gammas = [1.0]"""
        best_test_acc, best_train_acc, best_weights, best_sigma, best_gamma, best_sigma0 = -1, -1, None, -1, -1, -1
        for gamma, sigma, sigma0 in product(gammas, sigmas, sigma0s):
            print("Gamma: ", gamma, "Sigma: ", sigma, "Sigma0: ", sigma0)
            hungarian_weights , assignment = pdm_multilayer_group_descent(
                batch_weights, sigma0_layers=sigma0, sigma_layers=sigma, batch_frequencies=batch_freqs, it=0,
                gamma_layers=gamma
            )
            #print("hungarian weights",hungarian_weights)
            with open(args.data_dir+"\hungarian_weights\hungarian_weights_"+str(gamma)+"_"+str(sigma)+"_"+str(sigma0)+".txt", "w") as output:
                output.write(str(hungarian_weights))
            #train_dataset, test_dataset = get_dataset(args)

            for idx in idxs_users:
                idxs = (list(user_groups[idx]))
                idxs_train = list(idxs[:int(0.8 * len(idxs))])
                idxs_test = list(idxs[int(0.8 * len(idxs)):])
                print("indexes",idx,"test",len(idxs_test),"train",len(idxs_train))

                tr_dataset, _ = get_dataset(args)

                train_dataset = torch.utils.data.DataLoader(DatasetSplit(tr_dataset, idxs_train),
                                         batch_size=args.local_batch_size, shuffle=True, drop_last=False)
                # maybe add num_workers
                test_dataset = torch.utils.data.DataLoader(DatasetSplit(tr_dataset, idxs_test),
                                        batch_size=args.local_batch_size, shuffle=False)
                '''
                train_dataset, validloader = get_train_valid_loader(args,
                                                                  valid_size=0.2,
                                                                  shuffle=True,
                                                                  pin_memory=False)
                test_dataset = get_test_loader(args,
                                             shuffle=True,
                                             pin_memory=False)'''
                train_acc, test_acc, _, _,nets = compute_pdm_net_accuracy(hungarian_weights, train_dataset, test_dataset, n_classes,cls_freqs)
                res = {}
                key = (sigma0, sigma, gamma)
                res[key] = {}
                """for k in key:
                we should discuss about it later on
                """
                res[key]['shapes'] = list(map(lambda x: x.shape, hungarian_weights))
                res[key]['train_accuracy'] = train_acc
                res[key]['test_accuracy'] = test_acc

                #print('Sigma0: %s. Sigma: %s. Shapes: %s, Accuracy: %f' %(
                    #str(sigma0), str(sigma), str(res[key]['shapes']), test_acc))

                if test_acc > best_test_acc:
                    best_test_acc = test_acc
                    best_train_acc = train_acc
                    best_weights = hungarian_weights
                    best_sigma = sigma
                    best_gamma = gamma
                    best_sigma0 = sigma0
                print("Based on test")
                print('Best sigma0: %f, Best sigma: %f, Best Gamma: %f, Best accuracy (Test): %f. Training acc: %f' % (
                    best_sigma0, best_sigma, best_gamma, best_test_acc, best_train_acc))

        print("Running Iterative PDM matching procedure")
        logging.debug("Running Iterative PDM matching procedure")

        """sigma0s = [1.0]
        sigmas = [1.0]
        gammas = [1.0]"""
        gamma = best_gamma
        sigma = best_sigma
        sigma0 = best_sigma0
        """for (sigma0, sigma, gamma) in product(sigma0s, sigmas, gammas):"""
        logging.debug("Parameter setting: sigma0 = %f, sigma = %f, gamma = %f" % (sigma0, sigma, gamma))

        iter_nets = copy.deepcopy(nets)
        assignment = None
        file_name = args.data_dir + '/global_rounds/{}_{}_{}_{}_{}_{}_{}_{}_alpha{}.txt'. \
            format(args.dataset, args.model, args.num_users, args.frac,
                   args.local_ep, args.local_batch_size, args.iid, args.balanced, args.alpha)
        with open(file_name, "w") as f:
            f.write("gamma" + "," + "sigma" + "," + "sigma0" + "," + "comm_round" + "," + "expepochs"
                    "," + "n_nets" + "," + "train_acc_train_net"+ "," + "test_acc_train_net" + " \n")
        # Run for communication rounds iterations
        for i, comm_round in enumerate(range(args.communication_rounds)):
            print(f'\n | Global Training Round : {comm_round + 1} |\n')

            it = 3

            iter_nets_list = list(iter_nets.values())

            net_weights_new, train_acc, test_acc, new_shape, assignment, hungarian_weights, \
            conf_matrix_train, conf_matrix_test = compute_iterative_pdm_matching(nets,
                iter_nets_list, train_dataset, test_dataset, cls_count, args.net_config[-1],
                sigma, sigma0, gamma, it, old_assignment=assignment
            )

            print("Communication: %d, Train acc: %f, Test acc: %f, Shapes: %s" % (
            comm_round, train_acc, test_acc, str(new_shape)))
            print('CENTRAL MODEL CONFUSION MATRIX')
            print('Train data confusion matrix: \n %s' % str(conf_matrix_train))
            print('Test data confusion matrix: \n %s' % str(conf_matrix_test))

            iter_nets = load_new_state(iter_nets, net_weights_new)

            expepochs = args.local_ep
            train_correct_sum, train_total_sum, test_correct_sum, test_total_sum=0,0,0,0
            # Train these networks again
            for net_id, net in iter_nets.items():
                dataidxs = list(user_groups[net_id])
                idxs_train = list(dataidxs[:int(0.8 * len(dataidxs))])
                idxs_test = list(dataidxs[int(0.8 * len(dataidxs)):])
                print("Training network %s. n_training: %d" % (str(net_id), len(dataidxs)))
                train_dataset = torch.utils.data.DataLoader(DatasetSplit(tr_dataset, idxs_train),
                                                            batch_size=args.local_batch_size, shuffle=True,
                                                            drop_last=False)
                # maybe add num_workers
                test_dataset = torch.utils.data.DataLoader(DatasetSplit(tr_dataset, idxs_test),
                                                           batch_size=args.local_batch_size, shuffle=False)
                #net_train_dl, net_test_dl = get_dataloader(args.dataset, args.datadir, 32, 32, dataidxs)
                train_correct_train_net,train_total_train_net,test_correct_train_net,test_total_train_net = train_net(net_id, net, train_dataset, test_dataset, expepochs, args)
                train_correct_sum += train_correct_train_net
                train_total_sum += train_total_train_net
                test_correct_sum += test_correct_train_net
                test_total_sum += test_total_train_net
            file_name = args.data_dir + '/global_rounds/{}_{}_{}_{}_{}_{}_{}_{}_alpha{}.txt'. \
                format(args.dataset, args.model, args.num_users, args.frac,
                       args.local_ep, args.local_batch_size, args.iid, args.balanced, args.alpha)
            train_acc_train_net = train_correct_sum/train_total_sum
            test_acc_train_net = test_correct_sum/test_total_sum
            with open(file_name, "a") as f:
                    f.write(str(gamma)+","+str(sigma)+","+str(sigma0)+","+str(comm_round)+","+str(expepochs)+
                            ","+str(n_nets)+","+str(train_acc_train_net)+"," +str(test_acc_train_net)+" \n")

            print('\n Total Run Time: {0:0.4f}'.format(time.time() - start_time))


    else:
        print("you did not choose a correct communication type")


