import torch
from torch import nn
import torchvision
import math
import torch.nn.functional as F
#from src.models import networks
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class resnet_model(nn.Module):
    def __init__(self, input_channels = 3, num_classes = 7,use_dropout = False):
        super(resnet_model, self).__init__()
        resnet = torchvision.models.resnet18(pretrained=False)  # pretrained ImageNet ResNet-101
        # Remove linear and pool layers (since we're not doing classification)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

        # Resize image to fixed size to allow input images of variable size
        self.adaptive_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=0.5)
        self.use_dropout = use_dropout
        print('dropout')
        print(self.use_dropout)
        self.linear = nn.Linear(512, num_classes)

    def forward(self, images):
        """
        Forward propagation.

        :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
        :return: encoded images
        """
        batch = images.shape[0]
        out = self.resnet(images) 
        out = self.adaptive_pool(out)
#        out = F.avg_pool2d(out, 2)
#        out = out.view(out.size(0), -1)
        out = out.view(batch,-1)
        if self.use_dropout:
            out = self.dropout(out)
        out = self.linear(out)
#        out = out.permute(0, 2, 3, 1)  
        return out

    def fine_tune(self, fine_tune=True):
        """
        Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.

        :param fine_tune: Allow?
        """
        for p in self.resnet.parameters():
            p.requires_grad = False
        # If fine-tuning, only fine-tune convolutional blocks 2 through 4
        for c in list(self.resnet.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune
                

class ResidualBlock(nn.Module):
    """Residual Block."""
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Dropout(),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True))

    def forward(self, x):
        return x + self.main(x)



class Generator(nn.Module):
    """Generator. Encoder-Decoder Architecture."""
    def __init__(self, conv_dim=64, c_dim=7, repeat_num=6):
        super(Generator, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(3+c_dim, conv_dim, kernel_size=7, stride=1, padding=3, bias=False), nn.InstanceNorm2d(conv_dim, affine=True), nn.LeakyReLU(0.01, inplace=True))

        # Down-Sampling
        curr_dim = conv_dim
        self.conv2 = nn.Sequential(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False), nn.InstanceNorm2d(curr_dim*2, affine=True), nn.LeakyReLU(0.01, inplace=True))
        curr_dim = curr_dim * 2
        self.conv3 = nn.Sequential(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False), nn.InstanceNorm2d(curr_dim*2, affine=True), nn.LeakyReLU(0.01, inplace=True))
        curr_dim = curr_dim * 2
        
        # Bottleneck
        layers = []
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))
        self.main_res = nn.Sequential(*layers)
        
        # Up-Sampling
        self.dconv3 = nn.Sequential(nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False), nn.InstanceNorm2d(curr_dim//2, affine=True), nn.LeakyReLU(0.01, inplace=True))
        curr_dim = curr_dim // 2
        self.dconv2 = nn.Sequential(nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False), nn.InstanceNorm2d(curr_dim//2, affine=True), nn.LeakyReLU(0.01, inplace=True))
        curr_dim = curr_dim // 2

        self.dconv1 = nn.Sequential(nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False), nn.Tanh())

    def forward(self, x, c):
        # replicate spatially and concatenate domain information
        c = c.unsqueeze(2).unsqueeze(3)
        c = c.expand(c.size(0), c.size(1), x.size(2), x.size(3))
        xc = torch.cat([x, c], dim=1)

        h1 = self.conv1(xc)
        h2 = self.conv2(h1)
        h3 = self.conv3(h2)
        h4 = self.main_res(h3) + h3
        h5 = self.dconv3(h4) + h2
        h6 = self.dconv2(h5) + h1

        return self.dconv1(h6) + x


class Discriminator(nn.Module):
    """Discriminator. PatchGAN."""
    def __init__(self, image_size=128, conv_dim=64, c_dim=7, repeat_num=6):
        super(Discriminator, self).__init__()

        self.repeat_num = repeat_num

        layers = []
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01, inplace=True))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01, inplace=True))
            curr_dim = curr_dim * 2

        k_size = int(image_size / np.power(2, repeat_num))
        self.main = nn.Sequential(*layers)
        self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(curr_dim, c_dim, kernel_size=k_size, bias=False)

    def forward(self, x):
        h = x
        out_feats = []
        for i in range(0, self.repeat_num):
            h = nn.Sequential(*list(self.main.children())[i*2:(i+1)*2])(h)
            if i < 4:
                out_feats.append(h.squeeze())
        out_real = self.conv1(h)
        out_aux = self.conv2(h)
        return out_real.squeeze(), out_aux.squeeze(), out_feats

         

class origin_ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, dim_in, dim_out):
        super(origin_ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True))

    def forward(self, x):
        return x + self.main(x)


class origin_Generator(nn.Module):
    """Generator network."""
    def __init__(self, conv_dim=64, c_dim=7, repeat_num=6):
        super(origin_Generator, self).__init__()

        layers = []
        layers.append(nn.Conv2d(3+c_dim, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True))
        layers.append(nn.ReLU(inplace=True))

        # Down-sampling layers.
        curr_dim = conv_dim
        for i in range(2):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim*2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        # Bottleneck layers.
        for i in range(repeat_num):
            layers.append(origin_ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

        # Up-sampling layers.
        for i in range(2):
            layers.append(nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim//2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2

        layers.append(nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.Tanh())
        self.main = nn.Sequential(*layers)

    def forward(self, x, c):
        # Replicate spatially and concatenate domain information.
        # Note that this type of label conditioning does not work at all if we use reflection padding in Conv2d.
        # This is because instance normalization ignores the shifting (or bias) effect.
        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, c], dim=1)
        return self.main(x)
       
        
class origin_Discriminator(nn.Module):
    """Discriminator network with PatchGAN."""
    def __init__(self, image_size=128, conv_dim=64, c_dim=7, repeat_num=6):
        super(origin_Discriminator, self).__init__()
        layers = []
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2

        kernel_size = int(image_size / np.power(2, repeat_num))
        self.main = nn.Sequential(*layers)
        self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(curr_dim, c_dim, kernel_size=kernel_size, bias=False)
        
    def forward(self, x):
        h = self.main(x)
        out_src = self.conv1(h)
        out_cls = self.conv2(h)
        return out_src, out_cls.view(out_cls.size(0), out_cls.size(1)), None
    
    
def main():
    img = torch.Tensor(10,3,48,48)
    resnet = resnet_model()
    a = resnet(img)

if __name__ == '__main__':
    main()
