import torch

x = torch.tensor([4.0], requires_grad=True)
y = torch.tensor([4.0], requires_grad=True)
z = torch.sqrt(x ** 2 + 4 * y) 
z.backward()
print(x.grad, y.grad)
