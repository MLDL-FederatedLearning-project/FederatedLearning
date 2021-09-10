# Federated Learning: where machine learning and data privacy can coexist
# Abstract
Nowadays the big amount of data created by mobile devices can be used to give a better experience to the users thanks to AI and ML. There are many problems correlated with data such as the privacy issue, their sensitiveness but also their huge size which have to be considered. One of the modern approaches used to handle data is Federated Learning [1]. It changes the previous idea of centralized data making them decentralized and avoiding many of the problems mentioned above. In this work we implement the most popular algorithm of Federated Learning, Federated Averaging, and we try to solve one of its biggest constraint related to the big amount of communication rounds with a new algorithm called Probabilistic Federated Neural Matching.

# Introduction
In the last years, more people use phones, tablets and other mobile devices as primary computing instruments. These devices increased their performances and integrated powerful sensors that collect a large amount of data (e.g. cameras, facial detection identity, sensor fingerprint). Collecting this big amount of data in centralized servers raised risks and responsibilities about privacy.
Federated Learning
Introduced in 2016 by Google, Federated Learning is a machine learning procedure that allows to get advantages from data without storing them on a central server. Central server coordinates all participating devices (users), but the learning task is solved on user’s devices.
Each client never uploads his local dataset to the server but computes an update to the current global model downloaded by the server and only local updates are transferred to the server. These local updates are aggregated by the server, i.e. averaging weights, and then a single consolidated and improved global model is sent back to the devices.

There are two types of Federated Learning:

•	Centralized federated learning: central server coordinates the different steps of algorithms and all the participating nodes during the learning process. The server selects the nodes at the beginning of the training process and aggregates the received model updates (weights).

•	Decentralized federated learning: In this architecture, nodes are able to coordinate themselves to obtain the global model. 
In federated learning, decentralized algorithms can reduce the high communication cost on the central server. 

Federated Learning is facing different kind of problems and challenges and we decide to focus on communication costs which are a bottleneck of the Federated approach. We try to decrease the number of rounds needed to train our model implementing a different algorithm called Probabilistic Federated Neural Matching (PFNM).

# Experiments
We simulate federated learning scenario using a standard dataset: CIFAR10.
First, we implement the baseline in which we divide the dataset into 40000 samples for the training and 10000 samples for the valid and 10000 for the test.
We use a CNN similar to LeNet5 which has two 5x5 64-channel convolution layers, each precedes a 2x2 max-pooling layer, followed by three fully-connected layers with 1600, 384 and 192 channels respectively and finally a softmax linear classifier. The optimizer used is SGD with weight decay set to 4x10−4, learning rate of 0.01 and momentum of 0.9 (all parameters were kept constant without decay for simplicity) and the criterion for predicting the loss is NLLLoss.

The first issue we have to face with is the presence of overfitting. We solve it with normalization, data augmentation and limiting the number of epochs to 15. 
Once we have a robust and efficient model for the baseline we move to the Federated Learning Averaging algorithm.
In the experiment we simulate a real-world scenario in which have 100 users and at each communication round with the server only a fraction of 0.1 of them is randomly chosen. We consider four different partitions of the dataset: IID balanced, IID unbalanced, non- IID balanced and non-IID unbalanced. For IID users have the same data distribution among 10 labels instead for non-IID we have a random distribution. Balanced contains 500 samples (400 for train and 100 for test) per user and unbalanced which contains a random number of samples for each user.
The first non-IID partition we implement is random so to better consider all the possible data distribution cases we use the Dirichlet distribution based on the α parameter. If α -> 0 each user has samples only from one label while for α-> +∞ the distribution tends to be IID. The values we choose for α  are 0.0, 0.05,0.10, 0.20, 0.50, 1,10,100. 
The number of communication rounds and local epoch used are respectively 250, 1.
The big constraint of these experiments is the high number of communication rounds we need to obtain values of accuracy and loss which are as close as possible to the one of the baseline. Following the work done by Yurochkin et al. we decide to implement the PFNM algorithm which uses a fewer number of communication rounds to obtain better values of accuracy and loss with FedAVG.
We empirically analyze the values of the effect of the three parameters on the accuracy for multilayer neural matching with a number of active users equals to 10 and batch size 32 and 64. The values used are:

gammas = [1.0, 1e-3, 50.0]

sigmas = [1.0, 0.1, 0.5]

sigma0s 	= 	[1.0, 10.0]

Then we use the best combination of values found to compute the final results with 50 communication rounds and 10 local epochs.
The partition of data is the same used for the previous experiment with FedAVG.



