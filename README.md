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
Federated Learning is facing different kind of problems and challenges and we decide to focus on communication costs which are a bottleneck of the Federated approach. We try to decrease the number of rounds needed to train our model implementing a different algorithm called Probabilistic Federated Neural Matching (PFNM).![image](https://user-images.githubusercontent.com/87315995/132876246-abd4be08-fd10-4d7c-a9c0-1f4d88380b1e.png)

