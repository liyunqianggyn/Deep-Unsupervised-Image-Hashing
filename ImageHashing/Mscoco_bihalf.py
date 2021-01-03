import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torchvision import transforms
from torch.autograd import Variable
import torchvision
import numpy as np
from PIL import Image
import os
import os.path
from cal_map_mult import calculate_top_map, calculate_map, compress


# Hyper Parameters
num_epochs = 150
batch_size = 32
epoch_lr_decrease = 60
learning_rate = 0.001
gamma = 6
encode_length = 16
num_classes = 80

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class MScoco(torch.utils.data.Dataset):

    def __init__(self, root,
                 transform=None, target_transform=None, train=True, database_bool=False):
        self.loader = default_loader
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        if train:
            self.base_folder = 'train_coco.txt'
        elif database_bool:
            self.base_folder = 'database_coco.txt'
        else:
            self.base_folder = 'test_coco.txt'

        self.train_data = []
        self.train_labels = []

        filename = os.path.join(self.root, self.base_folder)
        # fo = open(file, 'rb')

        with open(filename, 'r') as file_to_read:
            while True:
                lines = file_to_read.readline()
                # print lines.split()
                if not lines:
                    break
                pos_tmp = lines.split()[0]
                pos_tmp = pos_tmp[39:]                
                # print pos_tmp
                pos_tmp = os.path.join(self.root, pos_tmp)
                label_tmp = lines.split()[1:]
                self.train_data.append(pos_tmp)
                self.train_labels.append(label_tmp)
        self.train_data = np.array(self.train_data)
        # self.train_labels.reshape()
        self.train_labels = np.array(self.train_labels, dtype=np.float)
        self.train_labels.reshape((-1, num_classes))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """

        img, target = self.train_data[index], self.train_labels[index]
        img = self.loader(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.train_data)


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
        transforms.RandomHorizontalFlip(),
        transforms.Scale(256),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Dataset
    train_dataset = MScoco(root='./data/Mscoco/',
                            train=True,
                            transform=train_transform)

    test_dataset = MScoco(root='./data/Mscoco/',
                            train=False,
                            transform=test_transform)

    database_dataset = MScoco(root='./data/Mscoco/',
                            train=False,
                            transform=test_transform,
                            database_bool=True)


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

            result_map = calculate_top_map(qB=queryB, rB=retrievalB, queryL=queryL, retrievalL=retrievalL, topk=5000)
            print('--------mAP@5000: {}--------'.format(result_map))  


if __name__ == '__main__':
    main() 
        
