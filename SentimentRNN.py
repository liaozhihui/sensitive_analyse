import torch.nn as nn
import torch
import numpy as np
from datautils import DataSet,Train,stopwords,deal_padding
from torch.nn.utils.rnn import pad_packed_sequence
class SentimentRNN(nn.Module):

    def __init__(self,vocab_size,output_size,embedding_dim,hidden_dim,n_layers,bidrectional=True,drop_prob=0.5):

        super(SentimentRNN,self).__init__()

        self.output_size = output_size
        self.n_layer = n_layers
        self.hidden_dim = hidden_dim
        self.bidrectional = bidrectional

        self.embeding = nn.Embedding(vocab_size,embedding_dim)
        self.lstm = nn.LSTM(embedding_dim,hidden_dim,n_layers,dropout=drop_prob,
                            batch_first=True,bidirectional=bidrectional)
        self.dropout = nn.Dropout(0.3)

        if bidrectional:
            self.fc = nn.Linear(hidden_dim*2,output_size)
        else:
            self.fc = nn.Linear(hidden_dim,output_size)
        self.sig = nn.Sigmoid()

    def forward(self,x,hidden):
        batch_size = x.size(0)

        x = x.long()

        embeds = self.embeding(x)
        lstm_out , (hn,cn) = self.lstm(embeds,hidden)
        # lstm_out, _ = pad_packed_sequence(encoder_outputs_packed, batch_first=True)

        if self.bidrectional:
            lstm_out = lstm_out.contiguous().view(-1,self.hidden_dim*2)
        else:
            lstm_out = lstm_out.contiguous().view(-1,self.hidden_dim)

        out = self.dropout(hn.contiguous().view(-1,self.hidden_dim))

        out = self.fc(out)

        sig_out = self.sig(out)

        sig_out = sig_out.view(batch_size,-1)
        # sig_out = sig_out[:,-1]

        return sig_out,hidden

    def init_hidden(self,batch_size):
        weight = next(self.parameters()).data
        number = 1

        if self.bidrectional:
            number = 2
        if torch.cuda.is_available():
            hidden = (weight.new(self.n_layer*number,batch_size,self.hidden_dim).zero_().cuda(),
                      weight.new(self.n_layer*number,batch_size,self.hidden_dim).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layer * number, batch_size, self.hidden_dim).zero_(),
                      weight.new(self.n_layer * number, batch_size, self.hidden_dim).zero_())

        return hidden

word2index={}
with open("word2index.csv", "r") as f:
    for line in f.readlines():
        result = line.split(",")
        word2index[result[0]] = int(result[1].strip())
data = DataSet(word2index, stopwords,Train)
vocab_size = len(word2index)
train_len = len(Train)
output_size = 3
embedding_dim = 512
hidden_dim = 256
n_layers = 1
bidirectional = False
net = SentimentRNN(vocab_size,output_size,embedding_dim,hidden_dim,n_layers,bidirectional)


def train(net,batch_size,train_loader):
    lr = 0.001
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(),lr=lr)
    epochs = 32

    print_every = 100

    clip = 5

    if torch.cuda.is_available():
        net.cuda()
    net.train()

    for e in range(epochs):
        h = net.init_hidden(batch_size)
        counter = 0
        train_loader.indexs=[i for i in range(train_len)]

        for _ in range(train_len//batch_size):
            try:
                data = next(train_loader.get_batch_shuffle(batch_size))
            except StopIteration:
                break
            counter +=1
            if torch.cuda.is_available():
                inputs,labels = deal_padding(data)
                inputs,labels = inputs.cuda(),labels.cuda()
            h = tuple([each.data for each in h])

            net.zero_grad()

            output,h = net(inputs,h)
            loss = criterion(output,labels.long())
            loss.backward()
            nn.utils.clip_grad_norm(net.parameters(),clip)
            optimizer.step()

            if counter % 50 == 0:

                pre = output.argmax(dim=-1).long() == labels.long()
                pre = (pre.float().sum())/batch_size
                print("Epoch:{}/{}...".format(e+1,epochs),
                      "Step:{}...".format(counter),
                      "Loss:{:.6f}...".format(loss.item()),
                      "precision:{:.3f}".format(pre))




if __name__ == '__main__':
    train(net,16,data)
