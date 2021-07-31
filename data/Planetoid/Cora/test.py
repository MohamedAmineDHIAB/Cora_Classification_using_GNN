import numpy as np
import torch
import seaborn as sns
from matplotlib import pyplot as plt
import model
from get_data import get_data


def test(model, data):
    model.eval()
    out = model(data.x, data.edge_index)
    # Use the class with highest probability.
    pred = out.argmax(dim=1)
    # Check against ground-truth labels.
    test_correct = pred[data.test_mask] == data.y[data.test_mask]
    # Derive ratio of correct predictions.
    test_acc = int(test_correct.sum()) / int(data.test_mask.sum())
    return test_acc


if __name__ == "__main__":
    # Initialize model
    model = model.GCN(hidden_channels=16)

    # get data and model for  testing
    data = get_data()[0]
    model.load_state_dict(torch.load('./data/model.pth'))

    # get test accuracy

    sns.set_theme(style="whitegrid")
    test_acc = test(model, data)
    print('-'*50+f'\nTest Accuracy   :    {test_acc:.4f}\n'+'-'*50)
    pred = model(data.x, data.edge_index)
    sns.barplot(x=np.array(range(7)), y=pred[torch.argmin(torch.norm(pred,dim=1,p=2))].detach().cpu().numpy())
    plt.savefig('./figs/output_example.png')






