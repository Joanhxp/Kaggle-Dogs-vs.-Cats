# -*- coding: utf-8 -*-


import torch
from torch import nn
from torchvision import transforms, models
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import time
import os

batch_size = 32
lr = 1e-3
use_gpu = True
fix_param = True
max_epoch = 10
    

transforms_ = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(128),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
        ])

data_folder = {
        'train':
            ImageFolder('./data/train', transform=transforms_),
        'val':
            ImageFolder('./data/val', transform=transforms_)
        }

dataloader = {
        'train':
            DataLoader(data_folder['train'], batch_size=batch_size, shuffle=True),
        'val':
            DataLoader(data_folder['val'], batch_size=batch_size, shuffle=True)
        }

data_size = {
        'train': len(dataloader['train'].dataset),
        'val': len(dataloader['val'].dataset)
        }
    
device = torch.device('cuda' if use_gpu else 'cpu')

transfer_model = models.resnet18(pretrained=True)
#if fix_param:
#    for param in transfer_model.parameters():
#        param.requries_grad = False
in_features = transfer_model.fc.in_features
transfer_model.fc = nn.Linear(in_features, 2)
transfer_model.to(device)

if fix_param:
    optimizer = torch.optim.Adam(transfer_model.fc.parameters(), lr=lr)
else:
    optimizer = torch.optim.Adam(transfer_model.parameters(), lr=lr)
    
criterion = nn.CrossEntropyLoss()

for epoch in range(1, max_epoch+1):
    print(f'{epoch}/{max_epoch}')
    print('*' * 10)
    print('Train')
    running_loss = 0.0
    running_acc = 0.0
    prev_time = time.time()
    
    transfer_model.train()
    for i, data in enumerate(dataloader['train'], 1):
        img, label = data
        img = img.to(device)
        label = label.to(device)
        
        output = transfer_model(img)
        loss = criterion(output, label)
        _, pred = torch.max(output, 1)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * batch_size
        num_correct = torch.sum(pred == label)
        running_acc += num_correct.item()
        
        if i % 100 == 0:
            print(f'Loss: {running_loss / (i * batch_size):.4f}, Acc: {running_acc / (i * batch_size):.4f}')

    running_loss /= data_size['train']
    running_acc /= data_size['train']
    temp_time = time.time() - prev_time
    print(f'Loss: {running_loss:.4f}, Acc: {running_acc:.4f}, Time: {temp_time:.0f}')
    
    print('Validation')
    transfer_model.eval()
    eval_loss = 0.0
    eval_acc = 0.0
    for i, data in enumerate(dataloader['val'], 1):
        img, label = data
        img = img.cuda()
        label = label.cuda()
        output = transfer_model(img)
        _, pred = torch.max(output, 1)
        loss = criterion(output, label)
        eval_loss += loss.item() * batch_size
        num_correct = torch.sum(pred == label)
        eval_acc += num_correct.item()
    eval_loss /= data_size['val']
    eval_acc /= data_size['val']
    print(f'Loss: {eval_loss:.4f}, Acc: {eval_acc:.4f}')

print('Finish Training!')
os.makedirs('./model_save', exist_ok=True)
torch.save(transfer_model.state_dict(), './model_save/resnet18.pth')

