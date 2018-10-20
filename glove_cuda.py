import pickle
import collections
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

import sys
import argparse

parser = argparse.ArgumentParser(description='name2ved embedding trainer')
parser.add_argument('--input', type=str, default='', metavar='N',
                    help='input pickle file dir')
parser.add_argument('--output', type=str, default='', metavar='N',
                    help='output file dir')
args = parser.parse_args()
dic_dire=args.input
embed_dire=args.output


d = pickle.load(file = open(dic_dire,'rb'))
word_counter = collections.Counter()
n_occo=0
for k,v in d.items():
    vk=list(v.keys())
    word_counter.update(vk)
    n_occo+=len(vk)
#vocabulary = [pair[0] for pair in word_counter.most_common()[0:top_k]]
vocabulary = [pair for pair in word_counter]
vocab_size=len(vocabulary)
index_to_word_map = dict(enumerate(vocabulary))
word_to_index_map = dict([(index_to_word_map[index], index) for index in index_to_word_map])
pickle.dump(index_to_word_map,file=open(embed_dire+'itw.p','wb'))
pickle.dump(word_to_index_map,file=open(embed_dire+'wti.p','wb'))

def data_loader(batch_size=10):
    n=0
    w_i_list=[]
    w_j_list=[]
    oc_list=[]
    for word_i,dict_j in d.items():
        for word_j,cooc in dict_j.items():
            if word_i != word_j:
                w_i_list.append(word_to_index_map[word_i])
                w_j_list.append(word_to_index_map[word_j])
                oc_list.append(cooc)

                if (n>0) and (n%batch_size)==0:
                    yield [oc_list, w_i_list, w_j_list]
                    w_i_list=[]
                    w_j_list=[]
                    oc_list=[]
                n+=1



class Glove(nn.Module):
    def __init__(self, embedding_dim, vocab_size, batch_size):
        super(Glove, self).__init__()
        self.word_embeddings = None

        """
        Your code goes here.
        """
        self.w=nn.Embedding(vocab_size,embedding_dim)
        #self.w_t=nn.Embedding(vocab_size,embedding_dim)

        self.b_i = nn.Embedding(vocab_size, 1)
        #self.b_j = nn.Embedding(vocab_size, 1)
    def forward(self,data_i,data_j,counts,x_max,alpha):
        """
        And here.
        """
        def wf(x):
            if x>100:
                return 100.0**alpha
            return x**alpha
        def cf(x):
            if x>100:
                return 100.0
            return x
        # print(data_i)
        # print(data_j)
        v_i=self.w(data_i)
        v_j=self.w(data_j)
        bi=self.b_i(data_i).squeeze(1)
        bj=self.b_i(data_j).squeeze(1)

        weights=np.array(list(map(wf,counts)))
        counts=np.array(list(map(cf,counts)))

        # print(sum(weights))
        # print("-----")
        # print(weights)
        # print(v_i.data[0])
        # # print('j')
        # print(v_j.data[0])
        # print((v_i * v_j).sum(1) )
        # print(bi.data)
        # print(bj.data)
        # print(bj.size())
        # print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        counts=Variable(torch.from_numpy(np.array(counts)).cuda()).float()
        weights=Variable(torch.from_numpy(weights).cuda()).float()
        diff_pure=(v_i * v_j).sum(1) + bi + bj
        diff_s1=diff_pure- counts
        diff_s2=torch.pow(diff_s1,2)
        output=(diff_s2 * weights+5*torch.pow(diff_pure,2)).sum()




        # print((v_i * v_j).sum(1) )
        # print((v_i * v_j).sum(1)+bi + bj)
        # print(((v_i * v_j).sum(1) + bi + bj) - counts)
        # print(torch.pow(((v_i * v_j).sum(1) + bi + bj) - counts, 2))
        # print((torch.pow(((v_i * v_j).sum(1) + bi + bj)- counts, 2) * weights))
        # print(output.data[0])
        # print(output)
        return (output,diff_s2.sum(),diff_s1)


    def init_weights(self):
        """
        And here.
        """
        nn.init.uniform(self.w.weight)
        #nn.init.uniform(self.w_t.weight)
        nn.init.uniform(self.b_i.weight)
        #nn.init.uniform(self.b_j.weight)
        return
    def add_embeddings(self):
        """
        And here.

        Give W_emb = W + W^tilde
        """
        self.word_embeddings=self.w.weight.data.cpu().numpy()
        return self.word_embeddings

    def get_embeddings(self, index):
        if self.word_embeddings is None:
            self.add_embeddings()
        return self.word_embeddings[index, :]

    def get_bias(self):
        return (self.b_i.weight.data.cpu().numpy())

def training(batch_size, num_epochs, model, optim, xmax, alpha):
    step = 0
    epoch = 0
    losses = []
    #total_batches = int(len(training_set) / batch_size)

    while epoch <= num_epochs:
        loader=data_loader(batch_size)
        model.train()
        losses=[]
        diffs=[]
        diff_s1s=[]
        for (counts, words, co_words) in loader:
            words_var = Variable(torch.LongTensor(words).cuda())
            co_words_var = Variable(torch.LongTensor(co_words).cuda())

            model.zero_grad()

            """
            Your code goes here.
            """
            optimizer=optim
            outputs=model(words_var,co_words_var,counts,xmax,alpha)
            loss=outputs[0]
            diff=outputs[1]
            diff_s1=outputs[2]
            losses.append(loss.data[0]/batch_size)
            diffs.append(diff.data[0]/batch_size)
            diff_s1s.append(np.mean(np.abs(diff_s1.data.cpu().numpy())))
            loss.backward()
            optimizer.step()

            step+=1
            if step % 200 ==0:
                with open(embed_dire+"longlog.txt",'a') as file:
                    file.write(str( step/(n_occo/batch_size))+'\n')


        epoch += 1


        word_emebddings = model.add_embeddings()
        with open(embed_dire+"log.txt",'a') as file:
            file.write( "Epoch:"+str(epoch)+' '+str(np.mean(losses))+' '+str(np.mean(diffs))+' '+str(np.mean(diff_s1s))+'\n')


embedding_dim = 150
vocab_size = vocab_size
batch_size = 1600
learning_rate = 10**(-7)
num_epochs = 70
alpha = 1.5
xmax = 100

glove = Glove(embedding_dim, vocab_size, batch_size).cuda()
glove.init_weights()
#optimizer = torch.optim.Adadelta(glove.parameters(), lr=1)
optimizer= torch.optim.SGD(glove.parameters(), lr=learning_rate, momentum=0)
#data_iter = batch_iter(nonzero_pairs, cooccurrences, batch_size)

training(batch_size, num_epochs, glove, optimizer, xmax, alpha)
np.savetxt(embed_dire+'embedding.txt',glove.add_embeddings())
b_i=glove.get_bias()
np.savetxt(embed_dire+'bias_i.txt',b_i)
#np.savetxt(embed_dire+'bias_j.txt',b_j)
