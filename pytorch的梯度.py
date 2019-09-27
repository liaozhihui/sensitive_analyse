import torch
from torch import nn
x= torch.ones(1)
w= torch.ones(1,requires_grad=True)
y = x*w
print(x.requires_grad,w.requires_grad,y.requires_grad)

x = torch.tensor(1., requires_grad=True)
w = torch.tensor(2., requires_grad=True)
b = torch.tensor(3., requires_grad=True)

y = w * x + b    # y = 2 * x + 3

# Compute gradients.
y.backward()#先要计算反向传播才有之后的x.grad的梯度值，否则x.grad为None

x=torch.randn(10,3)#input
y=torch.randn(10,2)#output
#建立一个全连接层
linear=nn.Linear(3,2)#初始的w,b都是随机的
print(linear.weight,linear.bias)
#构造损失含税和优化器
criterion=nn.MSELoss()
optimizer=torch.optim.SGD(linear.parameters(),lr=0.01)#指定优化器优化的是全连接层的参数
#前向传播
pred=linear(x)
#计算loss
loss=criterion(pred,y)
print(loss.item())
#计算反向传播，此时还未进行梯度下降
loss.backward()
print ('dL/dw: ', linear.weight.grad) #输出梯度
print ('dL/db: ', linear.bias.grad)
#梯度下降
optimizer.step()
#参数更新之后，再一次进行预测
pred=linear(x)
loss=criterion(pred,y)
print(loss.item())#经过一次梯度下降之后的预测loss
optimizer.zero_grad()
loss.backward()
print ('dL/dw2: ', linear.weight.grad) #输出梯度
print ('dL/db2: ', linear.bias.grad)
#梯度下降
optimizer.step()
#参数更新之后，再一次进行预测
pred=linear(x)
loss=criterion(pred,y)
print(loss.item())#经过一次梯度下降之后的预测loss


