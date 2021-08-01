# Cora_Classification_using_GNN


In this REPO we try to use Pytorch Geometric to build a Graph Neural Network and test it on the Cora Citation Dataset. Simple MLP models perform a lot worse than GNNs on this type of task, as the citation information is crucial for a correct classification therefore we use GNN (we can not just use the features of each node to classify it we need the relation ,i.e the citation, to get a better performance, which in GNN corresponds to the edges between nodes, things that you can not apply to simple MLP models)

## Cora :

The Cora dataset consists of 2708 scientific publications classified into one of seven classes. The citation network consists of 5429 links. Each publication in the dataset is described by a 0/1-valued word vector indicating the absence/presence of the corresponding word from the dictionary. The dictionary consists of 1433 unique words.


***source:*** <a href="https://relational.fit.cvut.cz/dataset/CORA">CORA</a>

## Description of the model:


- We have one large graph and not many individual graphs (like molecules)
- We infere on unlabeled nodes in this large graph and hence perform node-level predictions --> We have to use different nodes of the graph depending on what we want to do
- Nodes = Publications (Papers, Books ...)
- Edges = Citations
- Node Features = word vectors
- 7 Labels = Pubilcation type e.g. Neural_Networks, Rule_Learning, Reinforcement_Learning, 	Probabilistic_Methods...

We start by normalizing the features using torch geometric's transform functions.

Then we build our GNN which has the following structure:

`GCN(
  (conv1): GCNConv(1433, 16)
  (conv2): GCNConv(16, 16)
  (out): Linear(in_features=16, out_features=7, bias=True)
)`

we chose the following hyper-parameters for the model:

`learning_rate = 0.001
decay = 5e-5`

for the traning phase we get the following training loss plot:

![image](https://user-images.githubusercontent.com/85687148/127714992-d0041be9-ec7d-427a-ba70-8d7740790664.png)


We save our trained model after `10 000 epochs`
And then apply it on the test set and get the following accuracy:

`Test Accuracy   :    0.7710`

An example of output for the input features that causes the most confusion for which class to give is the following :

![image](https://user-images.githubusercontent.com/85687148/127743642-693eef06-92a0-46af-ad0d-07c3cb13a774.png)

The model predicts the class `0` since it has the highest probability, which means that the document is of type `Neural_Networks`.






