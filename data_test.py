import torch
data = torch.load("/home/miku/Code/DRAIL/expert_datasets/push_partial2.pt")
print(data.keys())
print(data['obs'].shape)
print(data['actions'].shape)