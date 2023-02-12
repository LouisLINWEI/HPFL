import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable


class LSTM_KWS(nn.Module):

    def __init__(self):
        super(LSTM_KWS, self).__init__()
        self.rnn = nn.LSTM(
#            input_size=28,
            input_size=10,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
        )
        self.out = nn.Linear(64,10)

    def forward(self,x):
        r_out, (h_n, h_c) = self.rnn(x, None)
        out = self.out(r_out[:,-1,:])
        return out

class LSTM_NLP(nn.Module):

    def __init__(self):
        super(LSTM_NLP, self).__init__()
        self.rnn = nn.LSTM(
#            input_size=28,
            input_size=50,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
        )
        self.out = nn.Linear(64,4)

    def forward(self,x):
        x_sum = torch.sum(torch.abs(x), dim=-1) # [batch_size, seq_len]
        # print "x_sum", x_sum
        not_padding = torch.ge(x_sum, 1e-5) 
        # print "not_padding: ", not_padding
        # print not_padding[0]
        # print not_padding[1] 
        length = torch.sum(not_padding, dim=-1) # [batch_size]
        sorted_length, perm_idx = torch.sort(length, descending=True)
        recover_idx = torch.zeros(length.size(),  dtype=perm_idx.dtype)
        for i in range(perm_idx.size(0)):
            recover_idx[perm_idx[i]] = i
        
        sorted_x = x[perm_idx]

        # print "sorted_length: ", sorted_length
        # print("perm_idx: ", perm_idx)

        packed_inputs = pack_padded_sequence(sorted_x, sorted_length, batch_first=True)
        #r_out, (h_n, h_c) = self.rnn(x, None)
        packed_outputs, (h_n, h_c) = self.rnn(packed_inputs, None)
        # print "h_n.size(): ", h_n.size()
        h_n = h_n[-1][recover_idx]
        
        # print "h_n", h_n
        origin_length = sorted_length[recover_idx]
        # print "origin_length: ", origin_length
        # print "is equal?: ", origin_length == length

        #print(str(r_out[:,-1,:]))
        # out = self.out([:, -1, :])
        out = self.out(h_n)
        return out

class LSTM_HAR(nn.Module):

    def __init__(self):
        super(LSTM_HAR, self).__init__()

        self.n_layers = 2
        self.n_hidden = 32
        self.n_classes = 6
        self.drop_prob = 0.5
        self.n_input = 9

        self.lstm1 = nn.LSTM(9, 32, 2, dropout=0.5)
        self.lstm2 = nn.LSTM(32, 32, 2, dropout=0.5)
        self.fc = nn.Linear(32, 6)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, hidden):
        x = x.squeeze()
        x = x.permute(2, 0, 1)
        x, hidden1 = self.lstm1(x, hidden)
        for i in range(1):
            #x = F.relu(x)
            x, hidden2 = self.lstm2(x, hidden)
        x = self.dropout(x)
        out = x[-1]
        out = out.contiguous().view(-1, 32)
        out = self.fc(out)
        out = F.softmax(out)

        return out

    def init_hidden(self, batch_size):
        ''' Initialize hidden state'''
        # Create two new tensors with sizes n_layers x batch_size x n_hidden,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data
        # if (train_on_gpu):
        if (torch.cuda.is_available() ):
            hidden = (weight.new(2, 100, 32).zero_().cuda(),
                weight.new(2, 100, 32).zero_().cuda())
        else:
            hidden = (weight.new(2, batch_size, 32).zero_(),
                weight.new(2, batch_size, 32).zero_())

        return hidden