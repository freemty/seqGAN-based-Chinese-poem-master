import pandas as pd 
import pickle as pkl
import os
import re
import numpy as np
from collections import Counter
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences



with open('data/Poem.pkl','rb') as f:
    texts = pkl.load(f)

texts7 = [text[:-1] for text in texts if  len(text[:-1])%32 == 0 and len(text[:-1])%6 != 0]
print('7 word num : {}'.format(len(texts7)))

vocab_size = 5000
text_seq = []
for text in texts7:
    for i in range(len(text)//32):
        text_seq.append(text[i*32:(i+1)*32])

tokenizer = Tokenizer(4999, char_level=True , oov_token='<unk>')
tokenizer.fit_on_texts(text_seq)
count = tokenizer.word_counts
vocab = tokenizer.word_index
vocab = {k : vocab[k] for k in vocab if vocab[k] <= 4999}# 0处空出来了，所以要加一
encode_text = tokenizer.texts_to_sequences(text_seq)
data = {'vocab':vocab,'text':encode_text}
with open('data/data.pkl','wb') as f:
    pkl.dump(data,f)
print('Done')


    










