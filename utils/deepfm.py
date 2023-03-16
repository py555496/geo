import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import pandas as pd
# 加载数据集
from sklearn.datasets import fetch_california_housing     
boston = fetch_california_housing()
X, y = boston.data, boston.target
#Data Preprocess
#data = pd.read_csv('data.csv')
data = pd.read_excel('new_data.xlsx', sheet_name='Sheet6')
#data = pd.read_excel('sample_data.xlsx', sheet_name='Sheet1')

#去除id特征 市辖区 街道乡镇 因为是非数字特征，后期可以尝试转化成hot参与计算
#data = data.drop(columns=['id','dist','str'])
data = data.drop(columns=['dist','str'])

#去除数据无效的空置
print(data.shape)
data = data.dropna()
c_name = ['gymd','squd','kinded','primd','highd','hospd','clind','medicad','busd','subd','ndis']
c_name += ['malld','superd','foodd','popud','jobd','trader','amuser','healthr','edur','degree','year','volume','green','bedr','restr','area','direct','type','x','y','rent']
data.drop_duplicates(subset=c_name, keep = 'last', inplace=True)
print(data.shape)

y = data['rent']
X = data.drop(columns=['rent'])
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# 数据归一化

# 划分训练集和测试集
"""
train_size = int(len(X) * 0.8)
X_train, y_train = X[:train_size], y[:train_size]
X_test, y_test = X[train_size:], y[train_size:]
"""

# 定义DeepFM模型
class DeepFM(nn.Module):
    def __init__(self, feature_dim, embedding_dim, hidden_dim):
        super(DeepFM, self).__init__()
        
        # FM部分
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(feature_dim, embedding_dim)
        self.linear = nn.Linear(embedding_dim, 1)
        
        # DNN部分
        self.fc1 = nn.Linear(feature_dim * embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()

        # total dense         
        self.final_fc = nn.Linear(3, 1)
        
    def forward(self, x):
        # FM部分
        emb = self.embedding(x)
        linear_part = self.linear(torch.sum(emb, dim=1))
        interaction_part = 0.5 * torch.sum(torch.pow(torch.sum(emb, dim=1), 2) - torch.sum(torch.pow(emb, 2), dim=1), dim=1)
        
        # DNN部分
        dnn_input = emb.view(-1, x.shape[1] * self.embedding_dim)
        dnn_output = self.relu(self.fc1(dnn_input))
        dnn_output = self.fc2(dnn_output)
        
        # 合并FM和DNN的输出
        #print(linear_part, interaction_part.reshape(-1,1), dnn_output.squeeze().reshape(-1,1))
        #output = linear_part + interaction_part + dnn_output.squeeze()
        #return self.final_fc(torch.cat([linear_output, fm_output, deep_output], dim=1))
        return self.final_fc(torch.cat((linear_part, interaction_part.reshape(-1,1), dnn_output.squeeze().reshape(-1,1)), dim=1))
        
        #return output


# 训练模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DeepFM(X_train.shape[1], embedding_dim=32, hidden_dim=32).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters())
for epoch in range(50):
    running_loss = 0.0
    for i in tqdm(range(0, len(X_train), 32)):
        inputs = torch.tensor(X_train[i:i+32], dtype=torch.long).to(device)
        labels = torch.tensor(y_train[i:i+32].to_numpy(), dtype=torch.float32).to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels.unsqueeze(1))
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    print('Epoch %d loss: %.3f' % (epoch+1, running_loss/len(X_train)))

# 评估模型
with torch.no_grad():
    inputs = torch.tensor(X_test, dtype=torch.long).to(device)
    labels = torch.tensor(y_test.to_numpy(), dtype=torch.float32).to(device)
    outputs = model(inputs)
    print("label:", labels, "\tpred:", outputs)
    mse = criterion(outputs, labels.unsqueeze(1))
print('Mean Squared Error:', mse.item())
