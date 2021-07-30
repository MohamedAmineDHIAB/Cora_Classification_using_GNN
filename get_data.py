import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures


def get_data():
    dataset = Planetoid(root='data/Planetoid', name='Cora',
                        transform=NormalizeFeatures())
    return(dataset)


if __name__ == "__main__":
    print(f'CUDA version:{torch.version.cuda}')
    dataset = get_data()
    # Get some basic info about the dataset
    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of features: {dataset.num_features}')
    print(f'Number of classes: {dataset.num_classes}')
    print(50*'=')

    # There is only one graph in the dataset, use it as new data object
    data = dataset[0]

    # Gather some statistics about the graph.
    print(data)
    print(f'Number of nodes: {data.num_nodes}')
    print(f'Number of edges: {data.num_edges}')
    print(f'Number of training nodes: {data.train_mask.sum()}')
    print(
        f'Training node label rate: {int(data.train_mask.sum()) / data.num_nodes:.2f}')
    print(f'Is undirected: {data.is_undirected()}')
