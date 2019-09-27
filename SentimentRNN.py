import torch.nn as nn
import torch
import numpy as np

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
        lstm_out , hidden = self.lstm(embeds,hidden)

        if self.bidrectional:
            lstm_out = lstm_out.contiguous().view(-1,self.hidden_dim*2)
        else:
            lstm_out = lstm_out.contiguous().view(-1,self.hidden_dim)

        out = self.dropout(lstm_out)

        out = self.fc(out)

        sig_out = self.sig(out)

        sig_out = sig_out.view(batch_size,-1)
        sig_out = sig_out[:,-1]

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


vocab_size = 60000
output_size = 1
embedding_dim = 400
hidden_dim = 256
n_layers = 2
bidirectional = False
net = SentimentRNN(vocab_size,output_size,embedding_dim,hidden_dim,n_layers,bidirectional)
print(net)
print(net.init_hidden(32))

def train(net,batch_size,train_loader,valid_loader):
    lr = 0.001
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(),lr=lr)
    epochs = 4

    print_every = 100

    clip = 5

    if torch.cuda.is_available():
        net.cuda()
    net.train()

    for e in range(epochs):
        h = net.init_hidden(batch_size)
        counter = 0

        for inputs,labels in train_loader:
            counter +=1
            if torch.cuda.is_available():
                inputs,labels = inputs.cuda(),labels.cuda()
            h = tuple([each.data for each in h])

            net.zero_grad()

            output,h = net(inputs,h)
            loss = criterion(output.squeeze(),labels.float())
            loss.backward()
            nn.utils.clip_grad_norm(net.parameters(),clip)
            optimizer.step()

            if counter % print_every == 0:
                val_h = net.init_hidden(batch_size)
                val_losses = []
                net.eval()
                for inputs,labels in valid_loader:

                    val_h = tuple([each.data for  each in val_h])
                    if torch.cuda.is_available():
                        inputs,labels = inputs.cuda(),labels.cuda()

                    outpus,val_h = net(inputs,val_h)
                    val_loss = criterion(output.squeeze(),labels.float())
                    val_losses.append(val_loss.item())
                net.train()
                print("Epoch:{}/{}...".format(e+1,epochs),
                      "Step:{}...".format(counter),
                      "Loss:{:.6f}...".format(loss.item()),
                      "Val Loss:{:.6f}".format(np.mean(val_losses)))




