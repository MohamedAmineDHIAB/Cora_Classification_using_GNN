# Cora_Classification_using_GNN


In this REPO we try to use Pytorch Geometric to build a Graph Neural Network and test it on the Cora Citation Dataset.

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

We normalize the features using torch geometric's transform functions.
