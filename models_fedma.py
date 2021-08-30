import numpy as np
import torchvision.transforms as transforms
from options import args_parser
from dataset_split import get_dataset
from models import CNNContainer, FcNet
import torch
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F


def pdm_prepare_weights(net):
    #print("net ", net)
    weights = []

    layer_i = 1
    statedict = net.state_dict()
    #print("stated dict in preparartion of the weights",statedict.keys())
    net_weights = []
    while True:

        if ('fc%d.weight' % layer_i) not in statedict.keys():
            break

        layer_weight = statedict['fc%d.weight' % layer_i].numpy().T
        layer_bias = statedict['fc%d.bias' % layer_i].numpy()

        net_weights.extend([layer_weight, layer_bias])
        layer_i += 1

    weights.append(net_weights)

    return weights


def compute_pdm_matching_multilayer(models, train_dl, test_dl, cls_freqs, n_classes, sigma0=None, it=0, sigma=None, gamma=None):
    batch_weights = pdm_prepare_weights(models)
    batch_freqs = pdm_prepare_freq(cls_freqs, n_classes)
    res = {}
    best_test_acc, best_train_acc, best_weights, best_sigma, best_gamma, best_sigma0 = -1, -1, None, -1, -1, -1

    gammas = [1.0, 1e-3, 50.0] if gamma is None else [gamma]
    sigmas = [1.0, 0.1, 0.5] if sigma is None else [sigma]
    sigma0s = [1.0, 10.0] if sigma0 is None else [sigma0]

    for gamma, sigma, sigma0 in product(gammas, sigmas, sigma0s):
        print("Gamma: ", gamma, "Sigma: ", sigma, "Sigma0: ", sigma0)

        hungarian_weights = pdm_multilayer_group_descent(
            batch_weights, sigma0_layers=sigma0, sigma_layers=sigma, batch_frequencies=batch_freqs, it=it, gamma_layers=gamma
        )

        train_acc, test_acc, _, _ = compute_pdm_net_accuracy(hungarian_weights, train_dl, test_dl, n_classes,cls_freqs)

        key = (sigma0, sigma, gamma)
        res[key] = {}
        res[key]['shapes'] = list(map(lambda x: x.shape, hungarian_weights))
        res[key]['train_accuracy'] = train_acc
        res[key]['test_accuracy'] = test_acc

        print('Sigma0: %s. Sigma: %s. Shapes: %s, Accuracy: %f' % (
        str(sigma0), str(sigma), str(res[key]['shapes']), test_acc))

        if train_acc > best_train_acc:
            best_test_acc = test_acc
            best_train_acc = train_acc
            best_weights = hungarian_weights
            best_sigma = sigma
            best_gamma = gamma
            best_sigma0 = sigma0

    print('Best sigma0: %f, Best sigma: %f, Best Gamma: %f, Best accuracy (Test): %f. Training acc: %f' % (
    best_sigma0, best_sigma, best_gamma, best_test_acc, best_train_acc))

    return (best_sigma0, best_sigma, best_gamma, best_test_acc, best_train_acc, best_weights, res)


def pdm_prepare_freq(cls_freqs, n_classes):
    freqs = []

    for net_i in sorted(cls_freqs.keys()):
        net_freqs = [0] * n_classes

        for cls_i in cls_freqs[net_i]:
            net_freqs[cls_i] = cls_freqs[net_i][cls_i]

        freqs.append(np.array(net_freqs))

    return freqs


def match_layer(weights_bias, sigma_inv_layer, mean_prior, sigma_inv_prior, gamma, it):
    J = len(weights_bias)

    group_order = sorted(range(J), key=lambda x: -weights_bias[x].shape[0])

    batch_weights_norm = [w * s for w, s in zip(weights_bias, sigma_inv_layer)]
    prior_mean_norm = mean_prior * sigma_inv_prior

    global_weights = prior_mean_norm + batch_weights_norm[group_order[0]]
    global_sigmas = np.outer(np.ones(global_weights.shape[0]), sigma_inv_prior + sigma_inv_layer[group_order[0]])

    popularity_counts = [1] * global_weights.shape[0]

    assignment = [[] for _ in range(J)]

    assignment[group_order[0]] = list(range(global_weights.shape[0]))

    ## Initialize
    for j in group_order[1:]:
        global_weights, global_sigmas, popularity_counts, assignment_j = matching_upd_j(batch_weights_norm[j],
                                                                                        global_weights,
                                                                                        sigma_inv_layer[j],
                                                                                        global_sigmas, prior_mean_norm,
                                                                                        sigma_inv_prior,
                                                                                        popularity_counts, gamma, J)
        assignment[j] = assignment_j

    ## Iterate over groups
    for iteration in range(it):
        random_order = np.random.permutation(J)
        for j in random_order:  # random_order:
            to_delete = []
            ## Remove j
            Lj = len(assignment[j])
            for l, i in sorted(zip(range(Lj), assignment[j]), key=lambda x: -x[1]):
                popularity_counts[i] -= 1
                if popularity_counts[i] == 0:
                    del popularity_counts[i]
                    to_delete.append(i)
                    for j_clean in range(J):
                        for idx, l_ind in enumerate(assignment[j_clean]):
                            if i < l_ind and j_clean != j:
                                assignment[j_clean][idx] -= 1
                            elif i == l_ind and j_clean != j:
                                print('Warning - weird unmatching')
                else:
                    global_weights[i] = global_weights[i] - batch_weights_norm[j][l]
                    global_sigmas[i] -= sigma_inv_layer[j]

            global_weights = np.delete(global_weights, to_delete, axis=0)
            global_sigmas = np.delete(global_sigmas, to_delete, axis=0)

            ## Match j
            global_weights, global_sigmas, popularity_counts, assignment_j = matching_upd_j(batch_weights_norm[j],
                                                                                            global_weights,
                                                                                            sigma_inv_layer[j],
                                                                                            global_sigmas,
                                                                                            prior_mean_norm,
                                                                                            sigma_inv_prior,
                                                                                            popularity_counts, gamma, J)
            assignment[j] = assignment_j

    print('Number of global neurons is %d, gamma %f' % (global_weights.shape[0], gamma))

    return assignment, global_weights, global_sigmas


def process_softmax_bias(batch_weights, last_layer_const, sigma, sigma0):
    J = len(batch_weights)
    sigma_bias = sigma
    sigma0_bias = sigma0
    mu0_bias = 0.1
    softmax_bias = [batch_weights[j][-1] for j in range(J)]
    softmax_inv_sigma = [s / sigma_bias for s in last_layer_const]
    softmax_bias = sum([b * s for b, s in zip(softmax_bias, softmax_inv_sigma)]) + mu0_bias / sigma0_bias
    softmax_inv_sigma = 1 / sigma0_bias + sum(softmax_inv_sigma)
    return softmax_bias, softmax_inv_sigma


def patch_weights(w_j, L_next, assignment_j_c):
    if assignment_j_c is None:
        return w_j
    new_w_j = np.zeros((w_j.shape[0], L_next))
    new_w_j[:, assignment_j_c] = w_j
    return new_w_j


def pdm_multilayer_group_descent(batch_weights, batch_frequencies, sigma_layers, sigma0_layers, gamma_layers, it,
                        assignments_old=None):

    n_layers = int(len(batch_weights[0]) / 2)
    #print("batch weights", batch_weights, "len of that", len(batch_weights))
    J = len(batch_weights)
    D = batch_weights[0][0].shape[0]
    K = batch_weights[0][-1].shape[0]

    if assignments_old is None:
        assignments_old = (n_layers - 1) * [None]
    if type(sigma_layers) is not list:
        sigma_layers = (n_layers - 1) * [sigma_layers]
    if type(sigma0_layers) is not list:
        sigma0_layers = (n_layers - 1) * [sigma0_layers]
    if type(gamma_layers) is not list:
        gamma_layers = (n_layers - 1) * [gamma_layers]

    if batch_frequencies is None:
        last_layer_const = [np.ones(K) for _ in range(J)]
    else:
        last_layer_const = []
        total_freq = sum(batch_frequencies)
        #print("sum frequencies",total_freq)
        for f in batch_frequencies:
            #print("f constant",f)
            result = np.divide(f, total_freq, where=total_freq != 0)
            last_layer_const.append(result)
        print("last layer",last_layer_const)

    sigma_bias_layers = sigma_layers
    sigma0_bias_layers = sigma0_layers
    mu0 = 0.
    mu0_bias = 0.1
    assignment_c = [None for j in range(J)]
    L_next = None
    assignment_all = []

    ## Group descent for layer
    for c in range(1, n_layers)[::-1]:
        sigma = sigma_layers[c - 1]
        sigma_bias = sigma_bias_layers[c - 1]
        gamma = gamma_layers[c - 1]
        sigma0 = sigma0_layers[c - 1]
        sigma0_bias = sigma0_bias_layers[c - 1]
        if c == (n_layers - 1) and n_layers > 2:
            weights_bias = [np.hstack((batch_weights[j][c * 2 - 1].reshape(-1, 1), batch_weights[j][c * 2])) for j in
                            range(J)]
            sigma_inv_prior = np.array([1 / sigma0_bias] + (weights_bias[0].shape[1] - 1) * [1 / sigma0])
            mean_prior = np.array([mu0_bias] + (weights_bias[0].shape[1] - 1) * [mu0])
            sigma_inv_layer = [np.array([1 / sigma_bias] + [y / sigma for y in last_layer_const[j]]) for j in range(J)]
        elif c > 1:
            weights_bias = [np.hstack((batch_weights[j][c * 2 - 1].reshape(-1, 1),
                                       patch_weights(batch_weights[j][c * 2], L_next, assignment_c[j]))) for j in
                            range(J)]
            sigma_inv_prior = np.array([1 / sigma0_bias] + (weights_bias[0].shape[1] - 1) * [1 / sigma0])
            mean_prior = np.array([mu0_bias] + (weights_bias[0].shape[1] - 1) * [mu0])
            sigma_inv_layer = [np.array([1 / sigma_bias] + (weights_bias[j].shape[1] - 1) * [1 / sigma]) for j in
                               range(J)]
        else:
            weights_bias = [np.hstack((batch_weights[j][0].T, batch_weights[j][c * 2 - 1].reshape(-1, 1),
                                       patch_weights(batch_weights[j][c * 2], L_next, assignment_c[j]))) for j in
                            range(J)]
            sigma_inv_prior = np.array(
                D * [1 / sigma0] + [1 / sigma0_bias] + (weights_bias[0].shape[1] - 1 - D) * [1 / sigma0])
            mean_prior = np.array(D * [mu0] + [mu0_bias] + (weights_bias[0].shape[1] - 1 - D) * [mu0])
            if n_layers == 2:
                sigma_inv_layer = [
                    np.array(D * [1 / sigma] + [1 / sigma_bias] + [y / sigma for y in last_layer_const[j]]) for j in
                    range(J)]
            else:
                sigma_inv_layer = [
                    np.array(D * [1 / sigma] + [1 / sigma_bias] + (weights_bias[j].shape[1] - 1 - D) * [1 / sigma]) for
                    j in range(J)]

        assignment_c, global_weights_c, global_sigmas_c = match_layer(weights_bias, sigma_inv_layer, mean_prior,
                                                                      sigma_inv_prior, gamma, it)
        L_next = global_weights_c.shape[0]
        assignment_all = [assignment_c] + assignment_all

        if c == (n_layers - 1) and n_layers > 2:
            softmax_bias, softmax_inv_sigma = process_softmax_bias(batch_weights, last_layer_const, sigma, sigma0)
            global_weights_out = [global_weights_c[:, 0], global_weights_c[:, 1:], softmax_bias]
            global_inv_sigmas_out = [global_sigmas_c[:, 0], global_sigmas_c[:, 1:], softmax_inv_sigma]
        elif c > 1:
            global_weights_out = [global_weights_c[:, 0], global_weights_c[:, 1:]] + global_weights_out
            global_inv_sigmas_out = [global_sigmas_c[:, 0], global_sigmas_c[:, 1:]] + global_inv_sigmas_out
        else:
            if n_layers == 2:
                softmax_bias, softmax_inv_sigma = process_softmax_bias(batch_weights, last_layer_const, sigma, sigma0)
                global_weights_out = [softmax_bias]
                global_inv_sigmas_out = [softmax_inv_sigma]
            global_weights_out = [global_weights_c[:, :D].T, global_weights_c[:, D],
                                  global_weights_c[:, (D + 1):]] + global_weights_out
            global_inv_sigmas_out = [global_sigmas_c[:, :D].T, global_sigmas_c[:, D],
                                     global_sigmas_c[:, (D + 1):]] + global_inv_sigmas_out

    map_out = [g_w / g_s for g_w, g_s in zip(global_weights_out, global_inv_sigmas_out)]

    return map_out, assignment_all

def record_net_data_stats(y_train, net_dataidx_map):
    net_cls_counts = {}
    print("Y train",y_train,"net data index map",net_dataidx_map)
    for net_i, dataidx in net_dataidx_map.items():
        print("net i",net_i,"data index",dataidx)
        unq, unq_cnt = np.unique(y_train[net_i], return_counts=True)
        counting = {}
        for each in dataidx:
            if y_train[each] in counting:
                x = int(counting[y_train[each]])
                counting[y_train[each]] = x + 1
            else:
                counting[y_train[each]] = 1
        print("counting",counting)
        #print("uniqueness", unq, "uniqueness count",unq_cnt)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp

    return net_cls_counts


def load_cifar10_data(datadir):

    args = args_parser()
    cifar10_train_ds,cifar10_test_ds = get_dataset(args)
    X_train, y_train = cifar10_train_ds.data, cifar10_train_ds.targets
    X_test, y_test = cifar10_test_ds.data, cifar10_test_ds.targets
    return (X_train, y_train, X_test, y_test)


def partition_data(train, test, n_nets, alpha=0.5):
    X_train, y_train, X_test, y_test = load_cifar10_data(train)
    n_train = X_train.shape[0]
    #print("n train",n_train)
    idxs = np.random.permutation(n_train)
    #print("indexes", idxs)
    batch_idxs = np.array_split(idxs, n_nets)
    #print("batch index",batch_idxs)
    net_dataidx_map = {i: batch_idxs[i] for i in range(n_nets)}
    #print("net data index", net_dataidx_map)

    traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map)
    print("train data cls count",traindata_cls_counts)
    return traindata_cls_counts


def prepare_weight_matrix(n_classes, weights: dict):
    weights_list = {}

    for net_i, cls_cnts in weights.items():
        cls = np.array(list(cls_cnts.keys()))
        cnts = np.array(list(cls_cnts.values()))
        weights_list[net_i] = np.array([0] * n_classes, dtype=np.float32)
        weights_list[net_i][cls] = cnts
        weights_list[net_i] = torch.from_numpy(weights_list[net_i]).view(1, -1)

    return weights_list


def prepare_uniform_weights(n_classes, net_cnt, fill_val=1):
    weights_list = {}

    for net_i in range(net_cnt):
        temp = np.array([fill_val] * n_classes, dtype=np.float32)
        weights_list[net_i] = torch.from_numpy(temp).view(1, -1)
    return weights_list


def prepare_sanity_weights(n_classes, net_cnt):
    return prepare_uniform_weights(n_classes, net_cnt, fill_val=0)


def normalize_weights(weights):
    Z = np.array([])
    eps = 1e-6
    weights_norm = {}

    for _, weight in weights.items():
        if len(Z) == 0:
            Z = weight.data.numpy()
        else:
            Z = Z + weight.data.numpy()

    for mi, weight in weights.items():
        weights_norm[mi] = weight / torch.from_numpy(Z + eps)

    print("weights normalized",weights_norm)
    input()
    return weights_norm

"""ToDo :  Check the prediction which is 0 but it added to correct predictions"""
def get_weighted_average_pred(models: list, weights: dict, images,labels,optimizer, args=args_parser()):
    out_weighted = None
    criterion = torch.nn.NLLLoss()
    # Compute the predictions
    for model_i, model in enumerate(models):
        out = F.log_softmax(model(images), dim=-1)  # (N, C)
        if out_weighted is None:
            out_weighted = (out * weights[model_i])
        else:
            out_weighted += (out * weights[model_i])

    return out_weighted


def compute_ensemble_accuracy(models: list, dataloader, n_classes, train_cls_counts=None, uniform_weights=False, sanity_weights=False, args = args_parser()):

    correct, total = 0, 0
    true_labels_list, pred_labels_list = np.array([]), np.array([])

    was_training = [False]*len(models)
    for i, model in enumerate(models):
        if model.training:
            was_training[i] = True
            model.eval()
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                    momentum=args.momentum, weight_decay=args.weight_decay)
    if uniform_weights is True:
        weights_list = prepare_uniform_weights(n_classes, len(models))
    elif sanity_weights is True:
        weights_list = prepare_sanity_weights(n_classes, len(models))
    else:
        weights_list = prepare_weight_matrix(n_classes, train_cls_counts)
    print("weights list",weights_list)
    weights_norm = normalize_weights(weights_list)

    with torch.no_grad():
        for batch_idx, (images, target) in enumerate(dataloader):
            target = target.long()
            #print("models",models)
            #print("weights norm",weights_norm)
            out = get_weighted_average_pred(models, weights_norm, images, target, optimizer)
            #print("out",out)
            _, pred_label = torch.min(out, 1)
            pred_label = pred_label.view(-1)
            #print("pred label",pred_label)
            total += images.data.size()[0]
            correct += (pred_label == target.data).sum().item()
            pred_labels_list = np.append(pred_labels_list, pred_label.numpy())
            true_labels_list = np.append(true_labels_list, target.data.numpy())
            #print("correct", correct, "total", total, "batch index", batch_idx)
            print("pred label",pred_labels_list)
            #print("true label list", true_labels_list)


    print(correct, total)

    conf_matrix = confusion_matrix(true_labels_list, pred_labels_list)

    for i, model in enumerate(models):
        if was_training[i]:
            model.train()

    return correct / float(total), conf_matrix


def compute_pdm_net_accuracy(weights, train_dl, test_dl, n_classes,cls_freqs):

    dims = []
    x = np.array(weights[0])
    dims.append(x.shape[0])
    for i in range(0, len(weights), 2):
        x = np.array(weights[i])
        dims.append(x.shape[1])
    print("dims", dims)
    input_dim = dims[0]
    output_dim = dims[-1]
    hidden_dims = dims[1:-1]
    input_channel = 3
    num_filters = 64
    #num_filters = [weights[0].shape[0], weights[2].shape[0]]
    #print("num filter",num_filters)
    kernel_size = 5
    pdm_net = CNNContainer(input_channel,num_filters, kernel_size, input_dim, hidden_dims, output_dim)

    #pdm_net = FcNet(input_dim, hidden_dims, output_dim)
    statedict = pdm_net.state_dict()
    # print(pdm_net)

    i = 0
    layer_i = 0
    while i < len(weights):
        weight = weights[i]
        i += 1
        bias = weights[i]
        i += 1
        number_conv = layer_i+1
        statedict['fc%d.weight' % number_conv] = torch.from_numpy(weight.T)
        statedict['fc%d.bias' % number_conv] = torch.from_numpy(bias)
        """statedict['layers.%d.weight' % layer_i] = torch.from_numpy(weight.T)
        statedict['layers.%d.bias' % layer_i] = torch.from_numpy(bias)"""
        layer_i += 1
    #print("Statedict",statedict)
    pdm_net.load_state_dict(statedict)

    train_acc, conf_matrix_train = compute_ensemble_accuracy([pdm_net], train_dl, n_classes,train_cls_counts=cls_freqs ,uniform_weights=False,sanity_weights=False)
    test_acc, conf_matrix_test = compute_ensemble_accuracy([pdm_net], test_dl, n_classes,train_cls_counts=cls_freqs, uniform_weights=False,sanity_weights=False)

    return train_acc, test_acc, conf_matrix_train, conf_matrix_test

