import torch
from torch import nn
from torch.autograd import Variable
from torch import tensor
import torch.nn.functional as F
import pickle as pkl 

class highway(nn.Module):
    def __init__(self,input_size,output_size):
        super(highway , self).__init__()
        self.fc1 = nn.Linear(input_size,output_size)
        self.fc2 = nn.Linear(input_size,output_size)

    def forward(self,x):
        g = F.relu(self.fc1(x))
        t = F.sigmoid(self.fc2(x))
        out = g*t + (1. - t)*x
        return out

class Discriminator(nn.Module):

    def __init__(self,seq_len,vocab_size,embed_size,filter_sizes, num_filters,num_class,keep_prob,l2_lambda,start_token = 0,use_cuda = False):
        super(Discriminator,self).__init__()
        self.l2_lambda = l2_lambda
        self.use_cuda = use_cuda
        total_filters = sum(num_filters)

        self.embedding = nn.Embedding(vocab_size,embed_size)
        self.drop_out = nn.Dropout(p = keep_prob)
        self.convs = nn.ModuleList([nn.Conv2d(1, num_f, (f_size, embed_size)) for f_size, num_f in zip(filter_sizes, num_filters)
        ])
        self.highway = highway(total_filters,total_filters)
        self.fc = nn.Linear(total_filters,num_class)

    def forward(self,inputs):
        #1. Embedding Layer
        x = self.embedding(inputs).unsqueeze(1)
        #2. Convolution + maxpool layer for each filter size
        convs = [F.relu(conv(x)).squeeze(3) for conv in self.convs] 
        # batch_size * num_filter * seq_len
        pooled_out = [F.max_pool1d(conv, conv.size(2)).squeeze(2) for conv in convs]
        #3. Combine all the pooled features into a prediction
        pred = torch.cat(pooled_out, 1) # batch_size * sum(num_filters)
        #4. Add highway
        h_highway = self.highway(pred)
        #5. Add dropout. This is when feature should be extracted
        h_drop = self.drop_out(h_highway)
        #6. Final unnormalized scores and predictions
        score = self.fc(h_drop)
        #pred = F.log_softmax(self.fc(h_drop),dim = 1)
        return score
    
    def get_pred(self,inputs):
        #labels = abels.argmax()
        score = self(inputs)
        pred = F.softmax(score, dim =1)
        return pred
        
    def loss_fn(self,inputs,labels):
        NLL_loss = nn.NLLLoss()

        if self.use_cuda:
            inputs.cuda()
            labels.cuda()
        score = self(inputs)
        #print(pred)
        log_pred = F.log_softmax(score , dim = 1)
        l2 = self.l2_loss()
        loss = NLL_loss(log_pred , labels[:,0]) + l2
        return loss

    def l2_loss(self):
        W = self.fc.weight
        b = self.fc.bias
        l2 = torch.sum(W*W) + torch.sum(b*b)
        l2_loss = self.l2_lambda * l2
        return l2_loss


class D_loader(object):
    def __init__(self,pos_path,neg_path,use_cuda = False):
        self.use_cuda = use_cuda
        self.pos_path = pos_path
        self.neg_path = neg_path

    def create_batch(self,pos_num ,batch_size):
        self.pointer = 0
        with open(self.pos_path,'rb')as f:
            pos_sentence = pkl.load(f)
            index = torch.randperm(len(pos_sentence))[:pos_num]
            pos_sentence = pos_sentence[index]
        with open(self.neg_path,'rb')as f:
            neg_sentence = pkl.load(f)
        sentence = torch.cat((pos_sentence,neg_sentence),0)
        pos_label = [[1,0] for i in pos_sentence]
        neg_label = [[0,1] for i in neg_sentence]
        labels = torch.tensor(pos_label + neg_label)
        index = torch.randperm(len(labels))
        labels = labels[index]
        sentence = sentence[index]
        self.labels = torch.split(labels,batch_size)[:-1]#把最后残的一个batch剔除
        self.sentence = torch.split(sentence,batch_size)[:-1]
        self.batch_num = len(self.sentence)
        if self.use_cuda:
            self.sentence = self.sentence.cuda()
            self.labels = self.labels.cuda()

    def next_batch(self):
        ret = self.sentence[self.pointer], self.labels[self.pointer]
        self.pointer  = (self.pointer + 1) % self.batch_num
        return ret

    def reset_pointer(self):
        self.pointer = 0
