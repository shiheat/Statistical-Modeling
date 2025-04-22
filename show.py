# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 数据准备
def prepare_data(predictions, node_features):
    num_nodes = node_features.shape[0]
    edge_index = []
    edge_weight = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                flow = predictions[i, j]
                if flow > 0:
                    edge_index.append([i, j])
                    edge_weight.append(flow)

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(edge_weight, dtype=torch.float32)
    node_features = torch.tensor(node_features, dtype=torch.float32)

    data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_weight)
    return data

# 定义 GCN 模型
class GCNModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCNModel, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_weight):
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return F.log_softmax(x, dim=1)

# 训练模型
def train_model(model, data, epochs=100, lr=0.01):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.edge_attr)
        loss = F.nll_loss(out, torch.randint(0, 2, (data.num_nodes,)))
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')
    return model

# 图可视化
def visualize_graph(data):
    G = nx.DiGraph()
    for i in range(data.num_nodes):
        G.add_node(i)
    for i in range(data.edge_index.shape[1]):
        src = data.edge_index[0, i].item()
        dst = data.edge_index[1, i].item()
        weight = data.edge_attr[i].item()
        G.add_edge(src, dst, weight=weight)

    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos)
    nx.draw_networkx_edges(G, pos, edge_color='r')
    labels = {i: i for i in range(data.num_nodes)}
    nx.draw_networkx_labels(G, pos, labels)
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels)
    plt.show()

# 加载真实数据
def load_data(file_path):
    try:
        # 尝试使用 GBK 编码读取 CSV 文件（常见于中文环境）
        df = pd.read_csv(file_path, encoding='gbk')
    except UnicodeDecodeError:
        # 如果 GBK 失败，尝试其他编码
        print("GBK decoding failed. Trying 'latin1' encoding...")
        df = pd.read_csv(file_path, encoding='latin1')
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        raise

    # 打印前几行数据以便调试
    print("CSV Data Preview:")
    print(df.head())

    # 假设 CSV 包含节点特征和流动预测矩阵
    # 请根据实际 CSV 结构调整以下代码

    # 假设节点特征是某些列（替换为实际列名）
    feature_cols = ['feature1', 'feature2', 'feature3']  # 替换为实际特征列名
    if all(col in df.columns for col in feature_cols):
        node_features = df[feature_cols].to_numpy()
    else:
        # 如果没有特征，生成随机特征
        num_nodes = len(df)
        node_features = np.random.rand(num_nodes, 3)
        print("Warning: Feature columns not found. Using random features.")

    # 假设流动预测是一个方阵，存储在 CSV 中（例如，按节点对存储）
    if 'source' in df.columns and 'target' in df.columns and 'flow' in df.columns:
        num_nodes = max(df['source'].max(), df['target'].max()) + 1
        predictions = np.zeros((num_nodes, num_nodes))
        for _, row in df.iterrows():
            src, tgt, flow = int(row['source']), int(row['target']), row['flow']
            predictions[src, tgt] = flow
    else:
        # 如果没有流动数据，生成随机预测
        num_nodes = len(df)
        predictions = np.random.rand(num_nodes, num_nodes)
        print("Warning: Flow data not found. Using random predictions.")

    return predictions, node_features

# 主程序
if __name__ == "__main__":
    # 加载真实数据
    file_path = r"E:\chesk\Desktop\应用统计建模大赛\final\test.csv"
    predictions, node_features = load_data(file_path)

    # 准备数据
    data = prepare_data(predictions, node_features)

    # 定义模型
    in_channels = data.x.shape[1]
    hidden_channels = 16
    out_channels = 2
    model = GCNModel(in_channels, hidden_channels, out_channels)

    # 训练模型
    model = train_model(model, data)

    # 可视化图
    visualize_graph(data)