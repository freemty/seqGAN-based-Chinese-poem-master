import torch
from torch import nn
import torch.nn.functional as F



a = torch.tensor([[1.,2],[2,4]])
b = torch.tensor([0,1])
a1 = F.log_softmax(a , 1)
print(F.softmax(a))
print(a1)
loss1 = F.cross_entropy(a,b)


loss_fn = nn.NLLLoss()
loss2 = loss_fn(a1,b)
print(loss1,loss2)