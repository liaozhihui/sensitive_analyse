import pandas as pd
import os
import jieba
import re
import random
import torch
from torch.nn.utils.rnn import pack_padded_sequence


file = os.path.dirname(os.path.abspath(__file__))
train_data = pd.read_csv(file+"/data/Train/Train_DataSet.csv")
label_data = pd.read_csv(file+"/data/Train/Train_DataSet_Label.csv")
result=pd.merge(train_data,label_data,how="left",on='id')[['title',"label"]]
Train = result.dropna().values
corpus = train_data["content"].dropna()
stopwords=set([word.strip() for word in open(file+"/data/stopwords.txt")])
class Corpus(object):
    def __init__(self,corpus):
        self.word2index={"UNK":0}
        self.word2count=dict()
        for raw in corpus:
            if raw !=" " and raw != "nan" and len(raw)>3:

                for word in jieba.cut(re.sub(r'[A-Za-z0-9]|/d+','',raw.replace(r"\.+","").replace(u'\u3000', u'').replace(u'\xa0', u' '))):
                    if word not in stopwords and word!="" and word!=" ":
                        if word not in self.word2count:
                            self.word2count[word] = 1
                        else:
                            self.word2count[word] += 1

        for k,v in self.word2count.items():
            if v>=3 and k not in self.word2index and k.strip()!="":
                self.word2index[k]=len(self.word2index)
        with open('word2index.csv', 'a') as f:
            for key, value in self.word2index.items():
                f.write('{0},{1}\n'.format(key, value))


class StopShuffle(Exception):
    def __init__(self,msg):
        super(StopShuffle,self).__init__()
        self.msg = msg
    def _str_(self):
        return self.msg



class DataSet(object):
    def __init__(self, word2index, stopwords,data):
        self.word2index = word2index
        self.stopwords = stopwords
        self.data = self.index(data)
        self.indexs= [i for i in range(len(data))]

    def cut_word(self,sententce):
        wordlist=[word for word in jieba.cut(re.sub(r'[A-Za-z0-9]|/d+', '', sententce.replace(r"\.+", "").replace(u'\u3000', u'').replace(u'\xa0', u' '))) if word not in self.stopwords]
        indexlist = list(map(lambda x:self.word2index.get(x,1),wordlist))
        return indexlist

    def index(self,data):
        for raw in data:
            raw[0]=self.cut_word(raw[0])
        return data

    def get_batch_shuffle(self,batch_size):
        if len(self.indexs)<batch_size:
            raise StopIteration
        outindex=random.sample(self.indexs, batch_size)
        output = []
        for i in outindex:
            self.indexs.remove(i)
            output.append(self.data[i])
        yield output

def deal_padding(data,batch_first=True):
    # for raw in data:
    #     print(len(raw[0]),raw[1])
    data = sorted(data,key=lambda x:len(x[0]),reverse=True)
    max_len = len(data[0][0])
    zeros = torch.zeros(len(data),max_len)
    seq_len=[]
    y_lable=[]
    for i in range(len(data)):
        length = len(data[i][0])
        seq_len.append(length)
        y_lable.append(data[i][1])
        print(torch.FloatTensor(data[i][0]))
        zeros[i,0:length] = torch.IntTensor(data[i][0])
    print(seq_len)
    print(zeros)
    pack=pack_padded_sequence(zeros,seq_len,batch_first=True)
    print(pack)
    print(y_lable)
    return pack,torch.Tensor(y_lable)


if __name__ == '__main__':
    word2index={}
    with open("word2index.csv", "r") as f:
        for line in f.readlines():
            result = line.split(",")
            word2index[result[0]] = int(result[1].strip())
    data = DataSet(word2index, stopwords,Train)
    # for i in range(1000):
    #     print(next(data.get_batch_shuffle(1000)))
    result = next(data.get_batch_shuffle(10))
    deal_padding(result)



