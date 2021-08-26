from torch import nn
import torch.nn.functional as F
#from options_FedMA import add_fit_args



class CNN(nn.Module):
    def __init__(self, args):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 5)
        #self.norm = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 64, 5)
        #self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(64*5*5, 384)
        self.fc2 = nn.Linear(384, 192)
        self.fc3 = nn.Linear(192, args.num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        #x = self.norm(x)
        x = x.view(-1, 64*5*5)
        #x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

class CNNContainer(nn.Module):
    def __init__(self, input_channel,num_filters, kernel_size, input_dim, hidden_dims, output_dim):
        super(CNNContainer, self).__init__()
        self.conv1 = nn.Conv2d(input_channel, num_filters, kernel_size, 1) #3,64
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size, 1) #new number,64

        self.fc1 = nn.Linear(input_dim, hidden_dims[0]) #1600, 384
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])#384,192
        self.fc3 = nn.Linear(hidden_dims[1], output_dim)#192,10

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(x)
        x = x.view(-1, x.size()[1] * x.size()[2] * x.size()[3])
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x


