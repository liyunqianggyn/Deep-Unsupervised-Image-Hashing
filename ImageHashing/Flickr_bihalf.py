import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd import Variable
import torchvision
from cal_map_mult import calculate_top_map, calculate_map, compress
import data.flickr25k as flickr25k

# Hyper Parameters
num_epochs = 100
batch_size = 32
epoch_lr_decrease = 60
learning_rate = 0.0001
gamma = 6
encode_length = 16
num_classes = 24

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
    test_loader, train_loader, database_loader = flickr25k.load_data(root='/tudelft.net/staff-bulk/ewi/insy/VisionLab/yunqiangli/data/Flicker/flickr25k/',
                                                                                 num_query = 2000,
                                                                                 num_train = 5000,
                                                                                 batch_size = batch_size,
                                                                                 num_workers = 4,
                                                                                 )

    cnn = CNN(encode_length=encode_length)

    # Loss and Optimizer
    optimizer = torch.optim.SGD(cnn.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)

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
            retrievalB, retrievalL, queryB, queryL = compress(database_loader, test_loader, cnn, classes=num_classes)

            result_map = calculate_map(qB=queryB, rB=retrievalB, queryL=queryL, retrievalL=retrievalL)
            print('--------mAP@All: {}--------'.format(result_map))  


if __name__ == '__main__':
    main() 
        
