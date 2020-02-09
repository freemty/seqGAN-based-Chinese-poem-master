import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import pickle as pkl 


class Generator(nn.Module):
    def __init__(self,embed_size,hidden_size,vocab_size,num_layers,rnn_type = 'lstm',use_cuda = False,start_token = 0):

        super(Generator,self).__init__()

        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.use_cuda = use_cuda
        self.rnn_type = rnn_type
        self.start_token = start_token

        self.embedding = nn.Embedding(vocab_size,embed_size)
        if rnn_type == 'gru':
            self.rnn = nn.GRU(embed_size,hidden_size,num_layers,bidirectional=False)
        elif rnn_type == 'lstm':
            self.rnn = nn.LSTM(embed_size,hidden_size,num_layers,bidirectional=False)
        else:
            raise ValueError
        self.rnn2out = nn.Linear(hidden_size,vocab_size)


    def forward(self,inputs ,state):
        '''
        unroll RNN for one step
        input size: batch_size * word_num
        '''
        embed = self.embedding(inputs)
        output , new_state = self.rnn(embed , state)
        pred = self.rnn2out(output)
        return pred ,new_state

    def init_state(self,batch_size = 1):
        h = Variable(torch.zeros(1,batch_size,self.hidden_size))
        if self.use_cuda:
            h = h.cuda()
        return h
    def generate_sample(self,batch_size,seq_len):

        if self.rnn_type == 'gru':
            state = self.init_state(batch_size)
        elif self.rnn_type == 'lstm':
            state = [self.init_state(batch_size),
                     self.init_state(batch_size)]    
        output = Variable(torch.zeros(batch_size , dtype=torch.long))
        if self.use_cuda:
            output = output.cuda()

        outputs , preds = [] , []
        for i in range(seq_len):
            pred , state = self(output.unsqueeze(0),state)
            output = F.softmax(pred.squeeze(0))
            #print(output[38])
            output = torch.multinomial(output, 1).squeeze(1)#why???
            #print(output)
            preds.append(pred)
            outputs.append(output)
        outputs = torch.stack(outputs).permute(1,0)
        preds = torch.cat(preds , dim = 0).permute(1,0,2)
        return outputs , preds

    def get_negsample(self,neg_path,sample_num,seq_len):

        neg_sample , _  = self.generate_sample(sample_num,seq_len)
        with open(neg_path,'wb') as f:
            pkl.dump(neg_sample,f)

    def CE_loss(self,inputs,labels):
        '''
        cross_entropy loss for pretrain Generator
        '''
        #loss_fn = nn.NLLLoss()
        batch_size , seq_len = labels.size()
        inputs = inputs.permute(1,0)
        labels = labels.permute(1,0) #seq_len-1 * batch_size
        loss = 0
        if self.rnn_type == 'gru':
            state = self.init_state(batch_size)
        elif self.rnn_type == 'lstm':
            state = [self.init_state(batch_size),
                    self.init_state(batch_size)]

        for i in range(seq_len):
            x = inputs[i].unsqueeze(0)
            pred , state = self(x,state)
            label = labels[i]
            pred = pred.squeeze(0)
            loss += F.cross_entropy(pred,label,reduce=True)
            #F.cross_entropy == F.log_softmax + nn.NLLLoss()
            #loss += loss_fn(F.log_softmax(pred),label) 
        loss = loss / seq_len
        return loss
    
    def generate_step(self,x,given_num):#init start
        '''
        G_beta use this function to complete the seqence in every time step
        ,thus they can get reward for each token
        '''
        batch_size , seq_len = x.size()
        if self.rnn_type == 'gru':
            state = self.init_state(batch_size)
        elif self.rnn_type == 'lstm':
            state = [self.init_state(batch_size),
                    self.init_state(batch_size)]
        inputs = Variable(torch.zeros(given_num+1 , batch_size,dtype=torch.long))
        inputs[0] = self.start_token 
        inputs[1:] = x[:,:given_num].permute(1,0)

        for inputs_x in inputs:
            output , state = self(inputs_x.unsqueeze(0) , state)
        output = F.softmax(output.squeeze(0))
        output = torch.multinomial(output, 1).squeeze(1)
        outputs = []
        if self.use_cuda :
            output = output.cuda()
        for i in range(seq_len - given_num):
            output, state = self(output.unsqueeze(0),state)
            output = F.softmax(output.squeeze(0))
            output = torch.multinomial(output, 1).squeeze(1)
            outputs.append(output)
        
        outputs = torch.cat((inputs[1:],torch.stack(outputs)) , dim = 0).permute(1,0)
        return outputs



    def PG_loss(self,inputs,preds,rewards):
        '''
        Policy Gradient loss in Adversial training
        '''
        if self.use_cuda:
            inputs = inputs.cuda() #batch_size * seq_len
            preds = preds.cuda() #batch_size * seq_len * vocab_size
            rewards = rewards.cuda() # batch_size * seq_len
        batch_size , seq_len = inputs.size()
        CE_loss = F.cross_entropy(preds.reshape(-1,self.vocab_size) , inputs.reshape(-1) , reduce= False)
        PG_loss = torch.mean(CE_loss * rewards.reshape(-1))
        
        return PG_loss
        

class G_loader(object):
    def __init__(self,text,use_cuda = False):
        self.text = text
        self.use_cuda = use_cuda
    def create_batch(self, batch_size , start_token = 0):
        self.pointer = 0
        self.batch_num = int(len(self.text) / batch_size)
        self.text = torch.tensor(self.text[:batch_size * self.batch_num])
        index = torch.randperm(batch_size * self.batch_num)
        self.text = self.text[index]
        labels = Variable(self.text)
        inputs = Variable(torch.zeros(self.text.size() , dtype=torch.long))
        inputs[:,0] = int(start_token)
        inputs[:,1:] = labels[:,:-1]

        self.labels = torch.split(labels, batch_size, 0)
        self.inputs = torch.split(inputs, batch_size, 0)
        if self.use_cuda:
            self.labels = self.labels.cuda()
            self.inputs = self.inputs.cuda()
        

    def next_batch(self):
        _input , _label = self.inputs[self.pointer],self.labels[self.pointer]
        self.pointer = (self.pointer + 1) % self.batch_num
        return  _input , _label

    def reset_pointer(self):
        self.pointer = 0


