import argparse
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from GCNII_layer import GCNIIdenseConv
import math
from common_blocks import batch_norm
import os
from torch_geometric.utils import remove_self_loops, add_self_loops
import numpy as np
import random
from torch_geometric.data import Data
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

character = np.loadtxt('D:/PyCharm 2023.2.1/learn/ADGCN/data/merged_USTC_40.txt',
                       dtype=float)
# 读取数据为numpy数组，dtype可以根据你的数据类型进行调整
classtype = np.loadtxt('D:/PyCharm 2023.2.1/learn/ADGCN/data/label_20_40.txt', dtype=int)

# 将numpy数组转换为torch tensor
tensor_data = torch.tensor(character, dtype=torch.float32)  # 选择适当的dtype
tensor_classtype = torch.tensor(classtype, dtype=torch.long)  # 选择适当的dtype

filename = 'D:/PyCharm 2023.2.1/learn/ADGCN/graph/USTC_40.txt'

data2 = []

with open(filename, 'r') as file:
    for line in file:
        parts = line.split()  # 假设每行数据由空格分隔
        if len(parts) == 2:
            # 将数据转换成整数后添加到data列表
            data2.append([int(parts[0]), int(parts[1])])

# 将列表转换成numpy数组（张量），然后进行转置
tensor = np.array(data2).T  # 使用.T属性来转置数组

# 打印转置后的张量
# print(tensor)
USTC = Data(x=tensor_data, y=tensor_classtype, edge_index=tensor)
USTC.num_classes = torch.unique(USTC.y).size(0)
USTC.num_features = 784  # 设置特征数量为2000（根据你的示例）


def load_data(dataset="USTC"):
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data', dataset)
    if dataset in ['USTC']:
        data = USTC
        num_nodes = data.x.size(0)
        if not isinstance(data.edge_index, torch.Tensor):
            data.edge_index = torch.from_numpy(data.edge_index).long()

        edge_index, _ = remove_self_loops(data.edge_index)
        edge_index = add_self_loops(edge_index, num_nodes=num_nodes)
        if isinstance(edge_index, tuple):
            data.edge_index = edge_index[0]
        else:
            data.edge_index = edge_index
        # devide training validation and testing set

        train_mask = torch.zeros((num_nodes,), dtype=torch.bool)
        val_mask = torch.zeros((num_nodes,), dtype=torch.bool)
        test_mask = torch.zeros((num_nodes,), dtype=torch.bool)
        train_num = 10
        val_num = 5
        for i in range(20):  # number of labels
            index = (data.y == i).nonzero()[:, 0]
            perm = torch.randperm(index.size(0))
            train_mask[index[perm[:train_num]]] = 1
            val_mask[index[perm[train_num:(train_num + val_num)]]] = 1
            test_mask[index[perm[(train_num + val_num):]]] = 1
        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask
        return data
    else:
        raise Exception(f'the dataset of {dataset} has not been implemented')


def remove_feature(data, miss_rate):
    num_nodes = data.x.size(0)
    erasing_pool = torch.arange(num_nodes)[~data.train_mask]
    size = int(len(erasing_pool) * miss_rate)
    idx_erased = np.random.choice(erasing_pool, size=size, replace=False)
    x = data.x
    x[idx_erased] = 0.
    return x


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(0)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


seeds = [100, 200, 300, 400, 500]

parser = argparse.ArgumentParser()
parser.add_argument('--layer', type=int, default=2, help='Number of layers.')
parser.add_argument('--type_norm', type=str, default="None")
parser.add_argument('--miss_rate', type=float, default=0.)
args = parser.parse_args()

dataset = 'USTC'

data = load_data(dataset)
if args.miss_rate > 0.:
    data.x = remove_feature(data, args.miss_rate)

print(data.train_mask.sum())
print(data.val_mask.sum())
print(data.test_mask.sum())

###################hyperparameters
nlayer = args.layer
dropout = 0.6
alpha = 0.2
lamda = 0.5
hidden_dim = 64
weight_decay1 = 0.01
weight_decay2 = 5e-4
lr = 0.005
patience = 200

skip_weight = 0.005
type_norm = args.type_norm
num_features = 784
num_classes = 20
#####################

GConv = GCNIIdenseConv


class GCNII_model(torch.nn.Module):
    def __init__(self):
        super(GCNII_model, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.layers_bn = torch.nn.ModuleList([])
        self.convs.append(torch.nn.Linear(num_features, hidden_dim))
        self.type_norm = type_norm
        if self.type_norm in ['None']:
            skip_connect = False
        else:
            skip_connect = True
        for i in range(nlayer):
            self.convs.append(GConv(hidden_dim, hidden_dim))
            self.layers_bn.append(batch_norm(hidden_dim, self.type_norm, skip_connect, skip_weight))

        self.convs.append(torch.nn.Linear(hidden_dim, num_classes))

        self.reg_params = list(self.convs[1:-1].parameters())
        self.non_reg_params = list(self.convs[0:1].parameters()) + list(self.convs[-1:].parameters())
        self.non_reg_params += list(self.layers_bn[0:].parameters())

    def forward(self):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        _hidden = []
        x = F.dropout(x, dropout, training=self.training)
        x = self.convs[0](x)
        x = F.relu(x)
        _hidden.append(x)
        for i, con in enumerate(self.convs[1:-1]):
            x = F.dropout(x, dropout, training=self.training)
            beta = math.log(lamda / (i + 1) + 1)
            x = con(x, edge_index, alpha, _hidden[0], beta, edge_weight)
            x = self.layers_bn[i](x)
            x = F.relu(x)
        x = F.dropout(x, dropout, training=self.training)
        x = self.convs[-1](x)
        return F.log_softmax(x, dim=1)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
acc_test_list = []
precision_list = []
recall_list = []
f1_list = []
for seed in seeds:
    set_seed(seed)
    model, data = GCNII_model().to(device), data.to(device)
    optimizer = torch.optim.Adam([
        dict(params=model.reg_params, weight_decay=weight_decay1),
        dict(params=model.non_reg_params, weight_decay=weight_decay2)
    ], lr=lr)


    def train():
        model.train()
        optimizer.zero_grad()
        loss_train = F.nll_loss(model()[data.train_mask], data.y[data.train_mask])
        loss_train.backward()
        optimizer.step()
        return loss_train.item()


    @torch.no_grad()
    def test():
        model.eval()
        logits = model()
        preds = logits.argmax(dim=1)
        labels = data.y[data.test_mask].cpu()
        preds = preds[data.test_mask].cpu()

        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted', zero_division=1)

        accuracy = accuracy_score(labels, preds)

        return precision, recall, f1, accuracy


    best_val_loss = 9999999
    best_val_acc = 0.
    test_acc = 0
    bad_counter = 0
    best_epoch = 0
    for epoch in range(1, 3000):
        loss_tra = train()
        precision, recall, f1, accuracy = test()
        if loss_tra < best_val_loss:
            best_val_loss = loss_tra
            test_acc = accuracy
            bad_counter = 0
            best_epoch = epoch
        else:
            bad_counter += 1
        if epoch % 20 == 0:
            log = 'Epoch: {:03d}, Train loss: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}, Test acc: {:.4f}'
            print(log.format(epoch, loss_tra, precision, recall, f1, accuracy))
        if bad_counter == patience:
            break
    log = 'Best Epoch: {:03d}, Train loss: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}, Test acc: {:.4f}'
    print(log.format(best_epoch, best_val_loss, precision, recall, f1, test_acc))
    acc_test_list.append(test_acc)
    precision_list.append(precision)
    recall_list.append(recall)
    f1_list.append(f1)

print('Test accuracy of 5 seeds:', acc_test_list)
print('Average test accuracy and standard deviation:', np.mean(acc_test_list), np.std(acc_test_list))
print('Average precision of 5 seeds:', precision_list)
print('Average recall of 5 seeds:', recall_list)
print('Average F1 score of 5 seeds:', f1_list)
print('Average precision:', np.mean(precision_list))
print('Average recall:', np.mean(recall_list))
print('Average F1 score:', np.mean(f1_list))
