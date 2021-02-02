import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.autograd import Function
from utilscluster import plot_latent_variable3d
import numpy as np
from cal_map import calculate_map, compress


name = 'SignLayer'
num_epochs = 100
batch_size = 128
learning_rate = 1e-3
encode_length = 16     # hash code length
gamma = 0.2           # parameter γ


def to_img(x):
    x = x.view(x.size(0), 1, 28, 28)
    return x


def min_max_normalization(tensor, min_value, max_value):
    min_tensor = tensor.min()
    tensor = (tensor - min_tensor)
    max_tensor = tensor.max()
    tensor = tensor / max_tensor
    tensor = tensor * (max_value - min_value) + min_value
    return tensor


def tensor_round(tensor):
    return torch.round(tensor)


# sign layer
class hash(Function):
    @staticmethod
    def forward(ctx, input):
        B_code = torch.sign(input)
        ctx.save_for_backward(input, B_code) 
        # ctx.save_for_backward(input)
        return B_code

    @staticmethod
    def backward(ctx, grad_output):
        input, B_code = ctx.saved_tensors
        uu = input.clone()
        size = uu.size()
        extrgrad = (uu - B_code)/(size[0]*size[1])
        grad_output = grad_output + gamma*extrgrad

        return grad_output


def hash_layer(input):
    return hash.apply(input)


def adjust_learning_rate(optimizer, epoch):
	update_list = [60, 80]
	if epoch in update_list:
		for param_group in optimizer.param_groups:
			param_group['lr'] = param_group['lr'] * 0.1
        
class autoencoder(nn.Module):
    def __init__(self, encode_length):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 256),
            nn.ReLU(True),
            nn.Linear(256, encode_length)
            )       
        torch.manual_seed(1) 
        self.decoder = nn.Sequential(
            nn.Linear(encode_length, 256),
            nn.ReLU(True),
            nn.Linear(256, 28 * 28),
            nn.Sigmoid())

        
    def forward(self, x):
        h = self.encoder(x)
        b = hash_layer(h)
        x = self.decoder(b)
        return x, h, b


def main():

    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda tensor:min_max_normalization(tensor, 0, 1)),
        transforms.Lambda(lambda tensor:tensor_round(tensor))
    ])
    

    dataset = MNIST('./data', train=True, transform=img_transform, download=True)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    testset = MNIST('./data', train=False, transform=img_transform, download=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=True)
    
    # visualize the distributions of the continuous feature U over 5,000 images
    visuadata =  MNIST('./data', train=False, transform=img_transform, download=True)
    X = dataset.data
    L = np.array(dataset.targets)
    
    first = True
    
    for label in range(10):
        index = np.where(L == label)[0]
    
        N = index.shape[0]
        np.random.seed(0)
        perm = np.random.permutation(N)
        index = index[perm]
    
        data = X[index[0:500]]
        labels = L[index[0:500]]
        if first:
            visualization_L = labels
            visualization_data = data
        else:
            visualization_L = np.concatenate((visualization_L, labels))
            visualization_data = torch.cat((visualization_data, data))
    
    
        first = False
    
        visuadata.data = visualization_data
        visuadata.targets = visualization_L
    
    # Data Loader
    visualization_loader = DataLoader(dataset=visuadata,
                                            batch_size=batch_size,
                                            shuffle=False,
                                            num_workers = 0)      
        
        
    model = autoencoder(encode_length=encode_length)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=1e-5)

    
    for epoch in range(num_epochs):
        print('--------training epoch {}--------'.format(epoch))        
        adjust_learning_rate(optimizer, epoch)    
        
        # train the model using SGD        
        for i, (img, _) in enumerate(train_loader):   
            img = img.view(img.size(0), -1)
            img = Variable(img)
  
            # ===================forward=====================
            output, _, _= model(img)
            loss = criterion(output, img)   #BCE reconstruction loss 
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        # Test the Model using testset            
        if (epoch + 1) % 1== 0:       


            '''
            Calculate the mAP over test set            
            '''             

            retrievalB, retrievalL, queryB, queryL = compress(train_loader, testloader, model)            
            result_map = calculate_map(qB=queryB, rB=retrievalB, queryL=queryL, retrievalL=retrievalL)
            print('---{}_mAP: {}---'.format(name, result_map))  
            
          
            
            '''
            visulization of latent variable over 5,000 images
            In this setting, we set encode_length = 3            
            '''
            if encode_length ==3:
                z_buf = list([])
                label_buf = list([])
                for ii, (img, labelb) in enumerate(visualization_loader):
                    img = img.view(img.size(0), -1)
                    img = Variable(img)
                    # ===================forward=====================
                    _, qz, _ = model(img)        
                    z_buf.extend(qz.cpu().data.numpy())
                    label_buf.append(labelb)
                X = np.vstack(z_buf)
                Y = np.hstack(label_buf)
                plot_latent_variable3d(X, Y, epoch, name)   
                        
            
if __name__ == '__main__':
    main()
            
              
