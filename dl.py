import torch
from torch import nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torchvision
import torch.utils.data as Data
import pandas as pd
import json
import pickle
import numpy as np
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer
import argparse
from load_dataset import load_dataset
import pdb

class mydataset(Dataset):
    def __init__(self,filename='train.pkl',transform=torchvision.transforms.ToTensor()) :
        self.dataset=pd.read_pickle(filename)
        self.transform = transform
    def __len__(self):
        return len(self.dataset['X'])
    def __getitem__(self, index):
        fig=self.dataset['X'][int(index)]
        # fig=self.transform(fig)
        return fig,(self.dataset['y'][int(index)])
    def data(self):
        return self.transform(self.dataset['X'])
    def target(self):
        return self.dataset['y']

class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.rnn = nn.LSTM(
            input_size=361,
            hidden_size=200,
            num_layers=5,
            batch_first=True
        ),
        self.linear = nn.Sequential(
            nn.Linear(200, 128),
            nn.Linear(128, 64),
            nn.Linear(64,2),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        pdb.set_trace()
        r_out, (h_n, h_c) = self.rnn[0](x, None)
        out = self.linear(r_out)
        return out

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.hidden_dim = 200
        self.n_layers = 5
        self.rnn = nn.RNN(input_size=361, hidden_size=200, num_layers=5, batch_first=True)
        self.linear = nn.Sequential(
            nn.Linear(200, 128),
            nn.Linear(128, 64),
            nn.Linear(64,2),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x = x.unsqueeze(1)
        # hidden = self.init_hidden(batch_size)
        out, hidden = self.rnn(x, None)

        out = out.contiguous().view(-1, 200)
        out = self.linear(out)

        return out
    
    def init_hidden(self, batch_size):
        hidden = torch.zeros(batch_size, self.hidden_dim)
        return hidden



class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=3, kernel_size=2, stride=2)
        self.max_pool1 = nn.MaxPool1d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv1d(3, 10, 3, 2)
        self.max_pool2 = nn.MaxPool1d(3, 2)
        self.conv3 = nn.Conv1d(10, 20, 3, 2)

        self.linear = nn.Sequential(
            nn.Linear(20*10, 128),
            nn.Linear(128, 64),
            nn.Linear(64,2),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = self.max_pool1(x)
        x = F.relu(self.conv2(x))
        x = self.max_pool2(x)
        x = F.relu(self.conv3(x))

        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        return x



class TRANSFORMER(nn.Module):
    def __init__(self):
        super(TRANSFORMER, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=361, nhead=19)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers = 6)
        decoder_layer = nn.TransformerDecoderLayer(d_model=361, nhead=19)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        
        # self.transformer = nn.Transformer(d_model=361, nhead=19, batch_first=True)

        self.linear = nn.Sequential(
            nn.Linear(361, 128),
            nn.Linear(128, 64),
            nn.Linear(64,2),
            nn.Sigmoid()
        )

    def forward(self, x):
        # pdb.set_trace()
        out = self.transformer_encoder(x)
        out = self.transformer_decoder(x, out)
        out = self.linear(out)
        return out



if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, default=100)
    argparser.add_argument('--batch_size', type=int, default=200)
    argparser.add_argument('--lr', type=float, default=0.0005)
    argparser.add_argument('--model', type=str, default='word2vec.model')
    argparser.add_argument('--train_file', type=str, default='train.pkl')
    argparser.add_argument('--valid_file', type=str, default='valid.pkl')
    argparser.add_argument('--test_file', type=str, default='test.pkl')
    argparser.add_argument('--time', type=bool, default=True)
    argparser.add_argument('--model_type', type=str, default='CNN')
    argparser.add_argument('--valid', type=float, default=0.2)
    opt = argparser.parse_args()

    # load dataset
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    BEST_VALID_ACC = 0.0
    # trainDataset, validDataset, testDataset = load_dataset(opt.train_file, opt.test_file, opt.model, opt.time, opt.valid)
    train_data=mydataset(filename='train.pkl')
    valid_data=mydataset(filename='valid.pkl')
    test_data=mydataset(filename='test.pkl')

    train_loader = Data.DataLoader(dataset=train_data, batch_size=opt.batch_size, shuffle=True)
    valid_loader=Data.DataLoader(dataset=valid_data,batch_size=opt.batch_size, shuffle=True)
    test_loader=Data.DataLoader(dataset=test_data,batch_size=opt.batch_size, shuffle=True)


    assert opt.model_type in ["CNN", "RNN", "LSTM", "TRANSFORMER"]
    if opt.model_type == "CNN":
        model = CNN()

    if opt.model_type == "RNN":
        model = RNN()
    
    if opt.model_type == "LSTM":
        model = LSTM()

    if opt.model_type == "TRANSFORMER":
        model = TRANSFORMER()

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    loss_func = nn.CrossEntropyLoss()

    # pdb.set_trace()

    for epoch in range(opt.epoch):
        for step, (x, y) in enumerate(train_loader):
            b_x = x.type(torch.FloatTensor).to(device)
            b_y = y.to(device)
            output = model(b_x)
            loss = loss_func(output, b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        accuracy=0.0
        sum_data=0.0
        for valid_x,valid_y in valid_loader:
            valid_x=valid_x.type(torch.FloatTensor).to(device)
            valid_y=valid_y.to(device)
            valid_output = model(valid_x)
            # import pdb;pdb.set_trace()
            pred_y = torch.max(valid_output, 1)[1].to(device).data # move the computation in GPU

            accuracy += torch.sum(pred_y == valid_y).type(torch.FloatTensor) 
            sum_data+=valid_y.size(0)
        accuracy=accuracy/sum_data
        if accuracy>BEST_VALID_ACC:
            BEST_VALID_ACC=accuracy
            torch.save(model.state_dict(), "best_" + opt.model_type + ".model")
        print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.cpu().numpy(), '| valid accuracy: %.4f' % accuracy)

    print('Best valid accuracy: %.4f' %BEST_VALID_ACC)

    accuracy = 0.0
    sum_data = 0.0
    for test_x,test_y in test_loader:
        test_x=test_x.type(torch.FloatTensor).to(device)
        test_y=test_y.to(device)
        test_output = model(test_x)
        # import pdb;pdb.set_trace()
        pred_y = torch.max(test_output, 1)[1].to(device).data # move the computation in GPU

        accuracy += torch.sum(pred_y == test_y).type(torch.FloatTensor) 
        sum_data+=test_y.size(0)
    accuracy=accuracy/sum_data

    print('Test accuracy: %.4f' %accuracy)
