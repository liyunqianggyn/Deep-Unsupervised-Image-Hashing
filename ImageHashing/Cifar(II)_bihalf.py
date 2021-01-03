import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import torchvision.datasets as dsets
from torchvision import transforms
from torch.autograd import Variable
import torchvision
import numpy as np
from cal_map_single import calculate_top_map, calculate_map, compress


# Hyper Parameters
num_epochs = 300
batch_size = 32
epoch_lr_decrease = 120
learning_rate = 0.0001
gamma = 6
encode_length = 16
num_classes = 10

# Bi-half layer
class hash(Function):
    @staticmethod
    def forward(ctx, U):

        # Yunqiang for half and half (optimal transport)
        _, index = U.sort(0, descending=True)
        N, D = U.shape
        B_creat = torch.cat((torch.ones([int(N/2), D]), -torch.ones([N - int(N/2), D]))).cuda()    
        B = torch.zeros(U.shape).cuda().scatter_(0, index, B_creat)
        
        ctx.save_for_backward(U, B) 
        
        return B

    @staticmethod
    def backward(ctx, g):
        U, B = ctx.saved_tensors
        add_g = (U - B)/(B.numel())

        grad = g + gamma*add_g

        return grad


def hash_layer(input):
    return hash.apply(input)


class CNN(nn.Module):
    def __init__(self, encode_length):
        super(CNN, self).__init__()
        self.vgg = torchvision.models.vgg16(pretrained=True)
        self.vgg.classifier = nn.Sequential(*list(self.vgg.classifier.children())[:6])
        for param in self.vgg.parameters():
            param.requires_grad = False
        torch.manual_seed(0)        
        self.fc_encode = nn.Linear(4096, encode_length)


    def forward(self, x):
        x = self.vgg.features(x)
        x = x.view(x.size(0), -1)
        x = self.vgg.classifier(x)
        h = self.fc_encode(x)
        b = hash_layer(h)

        return x, h, b

def adjust_learning_rate(optimizer, epoch):
    lr = learning_rate * (0.1 ** (epoch // epoch_lr_decrease))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
            
def main():
    


    train_transform = transforms.Compose([
        transforms.Scale(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Scale(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Dataset
    train_dataset = dsets.CIFAR10(root='./data/Cifar/',
                                train=True,
                                transform=train_transform,
                                download=True)

    test_dataset = dsets.CIFAR10(root='./data/Cifar/',
                                train=False,
                                transform=test_transform)

    database_dataset = dsets.CIFAR10(root='./data/Cifar/',
                                    train=False,
                                    transform=test_transform)
    

    # Re-Construct training, query and database set
    X = train_dataset.data
    L = np.array(train_dataset.targets)

    X = np.concatenate((X, test_dataset.data))
    L = np.concatenate((L, np.array(test_dataset.targets)))

    first = True

    for label in range(10):
        index = np.where(L == label)[0]

        N = index.shape[0]
        np.random.seed(0)
        perm = np.random.permutation(N)
        index = index[perm]

        data = X[index[0:1000]]
        labels = L[index[0:1000]]
        if first:
            test_L = labels
            test_data = data
        else:
            test_L = np.concatenate((test_L, labels))
            test_data = np.concatenate((test_data, data))

        data = X[index[1000:6000]]
        labels = L[index[1000:6000]]
        if first:
            dataset_L = labels
            data_set = data
        else:
            dataset_L = np.concatenate((dataset_L, labels))
            data_set = np.concatenate((data_set, data))

        data = X[index[1000:1500]]
        labels = L[index[1000:1500]]
        if first:
            train_L = labels
            train_data = data
        else:
            train_L = np.concatenate((train_L, labels))
            train_data = np.concatenate((train_data, data))

        first = False

        train_dataset.data = train_data
        train_dataset.targets = train_L.astype(np.long)
        test_dataset.data = test_data
        test_dataset.targets =(test_L).astype(np.long)
        database_dataset.data = data_set
        database_dataset.targets = (dataset_L).astype(np.long)
        
    # Data Loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            num_workers=4)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            num_workers=4)

    database_loader = torch.utils.data.DataLoader(dataset=database_dataset,
                                                batch_size=batch_size,
                                                shuffle=True,
                                                num_workers=4)
   
   
  
    cnn = CNN(encode_length=encode_length)


    # Loss and Optimizer
    optimizer = torch.optim.SGD(cnn.fc_encode.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)


    # Train the Model
    for epoch in range(num_epochs):
        cnn.cuda().train()
        adjust_learning_rate(optimizer, epoch)
        for i, (images, labels) in enumerate(train_loader):
            images = Variable(images.cuda())
            labels = Variable(labels.cuda().long())

            # Forward + Backward + Optimize
            optimizer.zero_grad()
            x, _, b = cnn(images)

            target_b = F.cosine_similarity(b[:int(labels.size(0) / 2)], b[int(labels.size(0) / 2):])
            target_x = F.cosine_similarity(x[:int(labels.size(0) / 2)], x[int(labels.size(0) / 2):])
            loss = F.mse_loss(target_b, target_x)
            loss.backward()
            optimizer.step()
        
        # Test the Model
        if (epoch + 1) % 10 == 0:
            cnn.eval()
            retrievalB, retrievalL, queryB, queryL = compress(database_loader, test_loader, cnn)
            result_map = calculate_map(qB=queryB, rB=retrievalB, queryL=queryL, retrievalL=retrievalL)
            print('--------mAP@All: {}--------'.format(result_map))  

if __name__ == '__main__':
    main()
              
  
        
        

        
            
            
        
        

