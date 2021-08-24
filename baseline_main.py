from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
import random
from options import args_parser
from update import test_inference, validation
from models import CNN
from dataset_split import get_train_valid_loader
from dataset_split import get_test_loader

if __name__ == '__main__':
    args = args_parser()
    if args.gpu:
        torch.cuda.set_device(args.gpu)
    device = 'cuda' if args.gpu else 'cpu'

    SEED = args.seed
    BATCH_SIZE = args.batch_size
    DATA_DIR = args.data_dir

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    # load datasets
    trainloader, validloader = get_train_valid_loader(args,
                                                      valid_size=0.2,
                                                      shuffle=True,
                                                      pin_memory=False)
    testloader = get_test_loader(args,
                                 shuffle=True,
                                 pin_memory=False)

    # BUILD MODEL
    args.model = 'cnn'
    # Convolutional neural network
    global_model = CNN(args=args)

    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()
    print("global model",global_model)

    # Training
    # Set optimizer and criterion
    args.optimizer = 'sgd'
    optimizer = torch.optim.SGD(global_model.parameters(), lr=args.lr,
                                momentum=args.momentum, weight_decay=args.weight_decay)
    criterion = torch.nn.NLLLoss().to(device)

    correct = 0
    total = 0

    accuracy_train = 0
    accuracy_valid = 0

    epoch_accuracy_train = []
    epoch_accuracy_valid = []
    epoch_accuracy_test = []

    loss_valid = 0

    epoch_loss_train = []
    epoch_loss_valid = []
    epoch_loss_test  = []

    for epoch in range(args.epochs):
        batch_loss_train = []
        batch_acc_train = []
        for batch_idx, (images, labels) in enumerate(trainloader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = global_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)
            accuracy_train = 100 * correct / total

            batch_loss_train.append(loss.item())
            batch_acc_train.append(accuracy_train)

        loss_avg_train = sum(batch_loss_train) / len(batch_loss_train)
        acc_avg_train = sum(batch_acc_train) / len(batch_acc_train)

        epoch_loss_train.append(loss_avg_train)
        epoch_accuracy_train.append(acc_avg_train)

        print('Epoch:', epoch+1)
        print('Train accuracy : {:.2f}%'.format(acc_avg_train), 'Train loss : {:.4f}'.format(loss_avg_train))

        correct = 0
        total = 0

        accuracy_valid, loss_valid = validation(args, global_model, validloader)
        accuracy_test, loss_test = test_inference(args, global_model, testloader)
        epoch_loss_valid.append(loss_valid)
        epoch_accuracy_valid.append(accuracy_valid)
        epoch_loss_test.append(loss_test)
        epoch_accuracy_test.append(accuracy_test)

        print('Valid accuracy : {:.2f}%'.format(accuracy_valid), 'Valid loss : {:.4f}'.format(loss_valid))


    fig, (ax1, ax2) = plt.subplots(2)

    train_loss, = ax1.plot(range(len(epoch_loss_train)), epoch_loss_train)
    valid_loss, = ax1.plot(range(len(epoch_loss_valid)), epoch_loss_valid)
    test_loss, = ax1.plot(range(len(epoch_loss_test)), epoch_loss_test)
    ax1.legend([train_loss, valid_loss], ['train loss', 'valid loss', 'test loss'])
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    train_acc, = ax2.plot(range(len(epoch_accuracy_train)), epoch_accuracy_train)
    valid_acc, = ax2.plot(range(len(epoch_accuracy_valid)), epoch_accuracy_valid)
    test_acc, = ax2.plot(range(len(epoch_accuracy_test)), epoch_accuracy_test)
    ax2.legend([train_acc, valid_acc], ['train accuracy', 'valid accuracy', 'test accuracy'])
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")


    plt.savefig(
        'C:/Users/Oana Madalina Breban/Downloads/FederatedLearning-main/FederatedLearning-main/accuracy_loss_baseline.png'.format(args.dataset,
                                                                                                        args.model,
                                                                                                        args.epochs))
    # testing

    test_acc, test_loss = test_inference(args, global_model, testloader)

    print('Test on', len(testloader), 'samples')
    print("Test Accuracy: {:.2f}%".format(test_acc))
    print('Test loss:', test_loss)
