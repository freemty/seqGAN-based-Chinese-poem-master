import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
import time
import os 
import pickle as pkl
from utils import *
from Generator import *
from Discriminator import *
from rollout import *
import numpy as np 

epochs = 100
g_step = 1
d_step = 5
rollout_num = 16
update_rate = 0.95

data_path = 'data/encode.pkl'
G_path = 'save/Gen'
D_path = 'save/Dis'
seq_len = 32
cuda = False
#Generator
batch_size = 64
embed_size = 50
hidden_size = 128
vocab_size = 5000
num_layers = 1
max_norm = 5.0
#Discriminator
num_class = 2
pos_path = 'data/pos.pkl'
neg_path = 'data/neg.pkl'
pos_num = 2000
neg_num = 2000
dis_embedd_szie = 100
dis_filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
dis_num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]
dis_dropout_keep_prob = 0.75
dis_l2_reg_lambda = 0.2


def pretrain_generator(G,g_op,data_loader,epochs = 3):
    data_loader.create_batch(batch_size,)
    batch_num = data_loader.batch_num
    for epoch in range(epochs):
        start = time.time()
        restore(G,G_path)
        data_loader.reset_pointer()
        total_loss = 0
        for i in range(batch_num):
            g_op.zero_grad()
            _input , _label = data_loader.next_batch()
            #print(_input )
            #print(_label)
            loss = G.CE_loss(_input , _label)
            loss.backward()
            clip_grad_norm_(G.parameters(), max_norm=max_norm)
            g_op.step()

            total_loss += (loss/seq_len)
            if (i + 1) % 100 == 0:
                print('batch{} , loss = {:.2f} , total_cost {:.2f}s'.\
                format(i+1 ,(total_loss / 100).data.item(),time.time() - start))
                total_loss = 0
            if (i + 1) % 500 == 0:
                save(G,G_path)
        print('G pre_train epoch{} , cost {:.2f}s'.\
            format(epoch+1 , time.time() - start))
        print('Pretrain Done')
        

def pretrain_discriminator(D,d_loader,d_op,G,epochs = 50):
    for e in range(epochs):
        G.get_negsample(neg_path,neg_num,seq_len)
        d_loader.create_batch(pos_num,batch_size)
        for r in range(1):
            d_loader.reset_pointer()
            losses = 0
            for i in range(d_loader.batch_num):
                d_op.zero_grad()
                sentence , labels = d_loader.next_batch()
                loss = D.loss_fn(sentence , labels)
                losses += loss
                loss.backward()
                d_op.step()
                if (i+1)%10 == 0:
                    print('batch {} , loss = {:.2f}'.format(i+1,losses.data.item()/10))
                    #print('l2 {:.2f}'.format(l2s.data.item()/10))
                    losses = 0
            print('epoch{},round{} Done!'.format(e + 1,r + 1))
        save(D,D_path)
    print('Pretrain Done')

def Adveraisl_training(G,g_op,D,d_op,d_loader):
    
    G_beta = Generator(embed_size,hidden_size,vocab_size,num_layers,use_cuda = cuda)
    restore(G_beta,G_path)
    log = open('save/log.txt','w')
    print('start Adveraisl_training!')
    
    for i in range(epochs):
        #train G
        start = time.time()
        buffer = 'start epoch{}'.format(i + 1)
        print(buffer)
        log.write(buffer)
        for _ in range(g_step):
            g_op.zero_grad()
            #Frist generate a whole seqence
            samples , preds = G.generate_sample(batch_size , seq_len)
            #Then get reward step by step
            rewards = get_rewards(G_beta,D,samples,rollout_num)
            loss = G.PG_loss(samples,preds,rewards)
            buffer = 'reward = {:.3f} PGloss={:.3f}'.format((rewards.mean()).data.item(),(loss).data.item())
            print(buffer)
            log.write(buffer)
            loss.backward()
            clip_grad_norm_(G.parameters(), max_norm=max_norm)
            g_op.step()

        #updata G_bata
        updataG_beta(G_beta ,G ,update_rate)

        #teain D
        for _ in range(d_step):
            G.get_negsample(neg_path,neg_num,seq_len)
            d_loader.create_batch(pos_num,batch_size)
            for r in range(3):
                d_loader.reset_pointer()
                losses = 0
                for j in range(d_loader.batch_num):
                    d_op.zero_grad()
                    sentence , labels = d_loader.next_batch()
                    loss  = D.loss_fn(sentence , labels)
                    losses += loss
                    loss.backward()
                    d_op.step()
        loss = test(G,g_loader)
        buffer = 'test loss={:.3f}'.format((loss))
        log.write(buffer)
        print('epoch{}Done! cost{:.2f}s'.format(i + 1 , time.time() - start))
        save(D,D_path)
        save(G,G_path)

def get_rewards(G_beta, D, sample,rollout_num = 16):
    rewards = torch.zeros(seq_len,batch_size)
    for i in range(rollout_num):
        for given_num in range(1 , seq_len):
            MC_sample = G_beta.generate_step(sample , given_num)
            pred = D.get_pred(MC_sample)[:,0]
            rewards[given_num - 1] += pred
        pred = D.get_pred(sample)[:,0]
        rewards[-1] += pred
    rewards = (rewards / rollout_num).permute(1,0)
    #print(rewards)
    return rewards

def updataG_beta(G_beta,G,update_rate = 0.95):
        a = [g for g in G.parameters()]
        b = [gb for gb in G_beta.parameters()]
        for i in range(len(a)):
            if i == 0:
                b[i].data = a[i].data # embedding
            else:
                b[i].data = a[i].data * (1 - update_rate) + b[i].data * update_rate
    

if __name__ == "__main__":
    with open('data/data.pkl','rb')as f:
        data = pkl.load(f)
    voc2int = data['vocab']
    text = data['text']
    int2voc = data['int2voc']

    #Generator
    G = Generator(embed_size,hidden_size,vocab_size,num_layers,use_cuda = cuda)
    g_loader = G_loader(text,use_cuda = cuda)
    g_op = optim.Adam(G.parameters(),lr = 0.01)
    restore(G,G_path)

    #Discriminator
    D = Discriminator(seq_len,vocab_size,embed_size,dis_filter_sizes,dis_num_filters,num_class,dis_dropout_keep_prob,dis_l2_reg_lambda)
    d_loader = D_loader(pos_path,neg_path,use_cuda = cuda)
    d_op = optim.Adam(D.parameters(),lr = 1e-3)
    restore(D,D_path)

    #pretrain_generator(G,g_op,g_loader)
    #pretrain_discriminator(D,d_loader,d_op,G) 
    #Adveraisl_training(G,g_op,D,d_op,d_loader)
    #generate(G,D,100,seq_len,int2voc)