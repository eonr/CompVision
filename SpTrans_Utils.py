import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

plt.ion()

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32,3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 32, 3)
        self.conv3 = nn.Conv2d(32,64,3,padding=1)
        self.conv4 = nn.Conv2d(64,64,3,padding=1)        
        self.conv5 = nn.Conv2d(64,128,3,padding=1)
        self.bnorm = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(1152, 10)
     
        self.localization = nn.Sequential(
             nn.Conv2d(1, 5, kernel_size=5),
             nn.MaxPool2d(2, stride=2),
             nn.ReLU(True),
             nn.Conv2d(5, 8, kernel_size=5),
             nn.MaxPool2d(2, stride=2),
             nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
             nn.Linear(8 * 4 * 4, 32),
             nn.ReLU(True),
             nn.Linear(32, 3 * 2)
        )

        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0],dtype=torch.float)) 
        #Initializing theta with identity function
        
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 8 * 4 * 4)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x
    
    def forward(self, x):
        x = self.stn(x) 
                                                    
        x = (F.relu(self.conv1(x)))
        x = (F.relu(self.conv2(x)))
        x = self.pool(x)
        x = F.dropout(x,p=0.3)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = F.dropout(x,p=0.3)
        x = (F.relu(self.conv5(x)))
        x = self.pool(x)
        x = self.bnorm(x)
        x = x.view(-1, 1152)
        x = self.fc1(x)
        return x

#Visualization functions from the pytorch tutorial

def convert_image_np(inp):
    """Convert a Tensor to numpy image."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    return inp

def visualize_stn(loader):
    with torch.no_grad():
        # Get a batch of training data
        data = next(iter(loader))[0].to(device)

        input_tensor = data.cpu()
        transformed_input_tensor = model.stn(data).cpu()

        in_grid = convert_image_np(
            torchvision.utils.make_grid(input_tensor))

        out_grid = convert_image_np(
            torchvision.utils.make_grid(transformed_input_tensor))

        # Plot the results side-by-side
        f, axarr = plt.subplots(1, 2)
        axarr[0].imshow(in_grid)
        axarr[0].set_title('Dataset Images')

        axarr[1].imshow(out_grid)
        axarr[1].set_title('Transformed Images')

