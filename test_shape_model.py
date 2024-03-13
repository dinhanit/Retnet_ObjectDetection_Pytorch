from VisRetNet.models import RMT_S
import torch
net = RMT_S(num_classes=5)
print(net(torch.rand([1,3,224,224])).shape)