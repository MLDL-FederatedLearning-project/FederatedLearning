
import torch
from torch import nn
import numpy as np
from torch.utils.data import DataLoader, Dataset


class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)


class LocalUpdate(object):
    def __init__(self, args, dataset, idxs, logger):
        self.args = args
        self.logger = logger
        self.trainloader, self.testloader = self.train_test(
            dataset, list(idxs))
        self.device = 'cpu'
        # Default criterion set to NLL loss function
        self.criterion = nn.NLLLoss().to(self.device)

    def train_test(self, dataset, idxs):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """
        # split indexes for train, validation, and test (80, 10, 10)
        idxs_train = idxs[:int(0.8*len(idxs))]
        idxs_test = idxs[int(0.8*len(idxs)):]

        trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
                                 batch_size=self.args.local_batch_size, shuffle=True, drop_last = False)

        testloader = DataLoader(DatasetSplit(dataset, idxs_test),
                                batch_size=self.args.local_batch_size, shuffle=False)
        return trainloader, testloader

    def update_weights(self, model, global_round):
        # Set mode to train model
        model.train()
        epoch_loss = []

        # Set optimizer for the local updates
        self.args.optimizer = 'sgd'
        optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr, momentum=self.args.momentum, weight_decay=self.args.weight_decay)


        for iter in range(self.args.local_ep):
            batch_loss = []

            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)

                model.zero_grad()
                log_probs = model(images)
                loss = self.criterion(log_probs, labels)
                loss.backward()
                optimizer.step()

                print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.4f} '.format(
                        global_round+1, iter+1, len(images),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader), loss.item(), ))


                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
                epoch_loss.append(sum(batch_loss)/len(batch_loss))
            #print(model.state_dict())
        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def inference(self, model):
        """ Returns the inference accuracy and loss.
        """

        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        for batch_idx, (images, labels) in enumerate(self.testloader):
            images, labels = images.to(self.device), labels.to(self.device)

            # Inference
            outputs = model(images)
            batch_loss = self.criterion(outputs, labels)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            print("pred_labels", pred_labels)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        accuracy = 100*correct/total
        return accuracy, loss

batch_loss= []

def test_inference(args, model, testloader):
    """ Returns the test accuracy and loss.
    """

    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    device =  'cpu'
    criterion = nn.NLLLoss().to(device)
    #testloader = get_test_loader(args,
                    #shuffle=True,
                    #pin_memory=False)

    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)

        # Inference
        outputs = model(images)
        loss = criterion(outputs, labels)


        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)
    batch_loss.append(loss.item())
    accuracy = 100*correct/total
    loss= sum(batch_loss)/len(batch_loss)

    return accuracy, loss


def validation(args,model,validloader):
        """ Returns the inference accuracy and loss.
        """
        device = 'cpu'
        model.eval()
        criterion = nn.NLLLoss().to(device)
        loss, total, correct, loss_final = 0.0, 0.0, 0.0, 0.0
        batch_loss= []


        for batch_idx, (images, labels) in enumerate(validloader):
            images, labels = images.to(device), labels.to(device)

            # Inference
            with torch.no_grad():
                outputs = model(images)
                loss = criterion(outputs, labels)
                batch_loss.append(loss.item())


                # Prediction
                _, pred_labels = torch.max(outputs, 1)
                pred_labels = pred_labels.view(-1)
                correct += torch.sum(torch.eq(pred_labels, labels)).item()
                total += len(labels)

            val_epoch_loss = sum(batch_loss)/len(batch_loss)


        accuracy = 100 * correct / total

        return accuracy, val_epoch_loss
