import torch
import pickle as pkl


def restore(the_model , model_path):
    the_model.load_state_dict(torch.load(model_path))

def save(the_model , model_path):
    torch.save(the_model.state_dict(), model_path)


def updata_G_beta(G , G_beta , update_rate = 0.95):
    a = [g.data for g in G.parameters()]
    b = [gb.data for gb in G_beta.parameters()]
    for i in range(len(a)):
        b[i] = a[i] * update_rate + b[i] * 1
    return G_beta

def generate(G,D, batch_size , seq_len , int2vocab):
    samples , _ = G.generate_sample(batch_size,seq_len)
    texts = ''
    pred = D.get_pred(samples)
    print(pred)
    for i , encode in enumerate(samples):
        text = ""
        if pred[i , 0].data.item() > 0.7:
            for char in encode:
                text += int2vocab[char.data.item()]
            texts += (text+'\n')
    with open('save/sample.txt','w')as f:
        f.write(texts)

def test(G,g_loader , batch_num = 10):
    g_loader.create_batch(batch_size)
    loss = 0
    for i in range(batch_num):
        _input , _label = g_loader.next_batch()
        #print(_input )
        #print(_label)
        loss += G.CE_loss(_input , _label)
    print('test loss={:.3f}'.format((loss/batch_num).data.item()))
    return (loss/batch_num).data.item()



if __name__ == "__main__":
    with open('data/data.pkl','rb')as f:
        data = pkl.load(f)
    text = data['text']
    voc2int = data['vocab']
    int2voc =  {a[1] : a[0] for a in voc2int.items()}
    data['int2voc'] = int2voc
    with open('data/data.pkl','wb')as f:
        pkl.dump(data,f)
