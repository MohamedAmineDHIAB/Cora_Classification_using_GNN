import model
import torch
from get_data import get_data
import seaborn as sns
from matplotlib import pyplot as plt


def train():
    model.train()
    optimizer.zero_grad()
    # Use all data as input, because all nodes have node features
    out = model(data.x, data.edge_index)
    # Only use nodes with labels available for loss calculation --> mask
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss


if __name__ == '__main__':
    # Initialize model
    model = model.GCN(hidden_channels=16)

    # get data for traning and testing
    data = get_data()[0]

    # Use GPU
    device = torch.device("cuda:0")
    model = model.to(device)
    data = data.to(device)

    # Initialize Optimizer
    learning_rate = 0.001
    decay = 5e-5
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=learning_rate,
                                 weight_decay=decay)
    # Define loss function (CrossEntropyLoss for Classification Problems with
    # probability distributions)
    criterion = torch.nn.CrossEntropyLoss()

    losses = []
    for epoch in range(0, 10001):
        loss = train()
        losses.append(loss)
        if epoch % 1000 == 0:
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

    losses_float = [float(loss.cpu().detach().numpy()) for loss in losses]
    loss_indices = [i for i, l in enumerate(losses_float)]
    sns.lineplot(x=loss_indices, y=losses_float)
    plt.savefig('./figs/training_loss.png')
    torch.save(model.state_dict(), './data/model.pth')
