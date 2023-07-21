import torch
 
import torch
import torch.nn as nn
 
class g(nn.Module):
    def __init__(self):
        super(g, self).__init__()
        self.k = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1, padding=0, bias=False)
 
    def forward(self, z):
        return self.k(z)
    

 
class gg(nn.Module):
    def __init__(self,model):
        super(gg, self).__init__()
        #self.k = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1, padding=0, bias=False)
        self.model = model
    def forward(self, z):
        return self.model(z)
    

 
 
c = 2
h = 5
w = 5
z = torch.rand( (1,c , h , w)).float().view(1, c, h, w)*100
z.requires_grad = False
p = g()
for name,v in p.named_parameters():
    print(name,v) 
print("***********1**********")
k=gg(p)
optimizer=torch.optim.Adam(p.parameters(), lr=1)


r = k(z)
r = r.sum()
loss = (r - 1) * (r - 1)

optimizer.zero_grad()
loss.backward()
for name,v in p.named_parameters():
    print(name,v) 
print("**********2***********")
optimizer.step()
for name,v in p.named_parameters():
    print(name,v) 
print("**********3***********")