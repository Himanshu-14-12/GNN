# Node Classification with Graph Neural Networks

## Aim
A group of articles need to be divided into seven categories. We used the cora dataset to accomplish this. This database offers a network of article citations as well as a number of node attributes.

Before moving onto how to implement this, lets learn about the terminilogies used.
## Node Classification
Node Classification is a machine learning task in graph-based data analysis, where the goal is to assign labels to nodes in a graph based on the properties of nodes and the relationships between them.

Node Classification models aim to predict non-existing node properties (known as the target property) based on other node properties. Typical models used for node classification consists of a large family of graph neural networks. 

## Graph Neural Networks
Graph Neural Network is a type of Neural Network which directly operates on the Graph structure and provides an easy way to do node-level, edge-level, and graph-level prediction tasks.

## Why do we even use the graph structure - aren't the features enough?
Given that the citation information is essential for accurate classification, it appears that basic MLP models perform significantly worse than GNNs on this kind of job. It has been observed that it attains only a accuracy of 27% using the traditional measures along with Random Forest Classifier.

## Methodology
The GNN classification model follows the Design Space for Graph Neural Networks methodology as follows: 
* Preprocess the node features using a Feed Forward Network to produce initial node representations.
* Produce node embeddings by applying one or more skip-connected graph convolutional layers to the node representation.
* To create the final node embeddings, post-process the node embeddings using the Feed Forward Network.
* Use a Softmax layer to forecast the node class by feeding it the node embeddings.

## Observations
* Only have a relatively small set of training nodes (20 nodes per class)
* There are binary test and train masks of the size #nodes (0 - Test, 1 - Train)
* Dropout is only applied in the training step, but not for predictions
* Have 2 Message Passing Layers and one Linear output layer
* Use the softmax function for the classification problem
* The output of the model are 7 probabilities, one for each class

## Result 
Node features + GNN model => 71.4% acc.

## Future Works
The model can be improved further by using various other mixture of algorithms
* Cross-Validation
* Hyperparameter Optimization
* Different layer types GCN, GAT... 
* Including edge features 
* The best performance is currently at around 0.9 using Gradient Descent

## References
[Node Classification on CORA](https://paperswithcode.com/sota/node-classification-on-cora)
[Understanding Graph Neural Networks](https://www.youtube.com/watch?v=ABCGCf8cJOE)
[Node Classification with natural Gradient Descent](https://paperswithcode.com/paper/optimization-of-graph-neural-networks-with)
