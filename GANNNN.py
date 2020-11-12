import argparse
import torch
import torchvision
import torchvision.utils as vutils
import torch.nn as nn
from random import randint
from sklearn import preprocessing
import numpy as np
# 参数
parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=4)
parser.add_argument('--imageSize', type=int, default=64)
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--epoch', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.00005, help='learning rate, default=0.0002')
parser.add_argument('--beta', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--data_path', default='./face_data/', help='folder to train data')
parser.add_argument('--outf', default='./imgs/', help='folder to output images and model checkpoints')
opt = parser.parse_args()

# 显卡
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())

#数据
dataset = np.load('bearing_data.npy')
dataset=dataset.flatten()
dataset= preprocessing.StandardScaler().fit_transform(dataset.flatten()[:,np.newaxis])
dataset=dataset.reshape(-1,64,64)
dataset = dataset[:,np.newaxis,:,:]

dataset = torch.from_numpy(dataset).float()

dataloader = torch.utils.data.DataLoader(
    dataset=dataset,
    batch_size=opt.batchSize,
    shuffle=True,
    drop_last=True,
)

class NetG(nn.Module):
    def __init__(self, ngf, nz):
        super(NetG, self).__init__()
        # layer1输入的是一个100x1x1的随机噪声, 输出尺寸(ngf*8)x4x4
        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(inplace=True)
        )
        # layer2输出尺寸(ngf*4)x8x8
        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(inplace=True)
        )
        # layer3输出尺寸(ngf*2)x16x16
        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(inplace=True)
        )
        # layer4输出尺寸(ngf)x32x32
        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(inplace=True)
        )
        # layer5输出尺寸 1x64x64
        self.layer5 = nn.Sequential(
            nn.ConvTranspose2d(ngf, 1, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    # 定义NetG的前向传播
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        return out


# 定义鉴别器网络D
class NetD(nn.Module):
    def __init__(self, ndf):
        super(NetD, self).__init__()
        # layer1 输入 1 x 64 x 64, 输出 (ndf) x 32 x 32#64
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, ndf, kernel_size=5, stride=3, padding=1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # layer2 输出 (ndf*2) x 16 x 16
        self.layer2 = nn.Sequential(
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # layer3 输出 (ndf*4) x 8 x 8
        self.layer3 = nn.Sequential(
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # layer4 输出 (ndf*8) x 4 x 4
        self.layer4 = nn.Sequential(
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # layer5 输出一个数(概率)
        self.layer5 = nn.Sequential(
            nn.Conv2d(ndf * 8, 1, 3, 1, 1, bias=False),
            # nn.MaxPool2d(kernel_size=2),
        )
        self.layer6 = nn.Sequential(nn.Linear(4,1)
            ,nn.Sigmoid()
                                    )
    # 定义NetD的前向传播
    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = out.view(out.size(0), -1)
        out = self.layer6(out)
        return out

netG = NetG(opt.ngf, opt.nz).to(device)
netD = NetD(opt.ndf).to(device)

criterion = nn.BCELoss()
# optimizerG = torch.optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta, 0.999))
optimizerG = torch.optim.RMSprop(netG.parameters(), lr = opt.lr)
optimizerD = torch.optim.RMSprop(netD.parameters(), lr = opt.lr)
# optimizerD = torch.optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta, 0.999))

label = torch.FloatTensor(opt.batchSize)
real_label = 1
fake_label = 0

for epoch in range(1, opt.epoch + 1):
    for i, imgs in enumerate(dataloader):
        # 固定生成器G，训练鉴别器D
        optimizerD.zero_grad()
        ## 让D尽可能的把真图片判别为1
        imgs=imgs.to(device)
        output = netD(imgs)
        label.data.fill_(real_label)
        label=label.to(device)
        errD_real = criterion(output, label)
        errD_real.backward()
        ## 让D尽可能把假图片判别为0
        label.data.fill_(fake_label)
        noise = torch.randn(opt.batchSize, opt.nz, 1, 1)
        noise=noise.to(device)
        fake = netG(noise)  # 生成假图

        output = netD(fake.detach()) #避免梯度传到G，因为G不用更新
        errD_fake = criterion(output, label)
        errD_fake.backward()
        errD = errD_fake + errD_real
        optimizerD.step()

        # 固定鉴别器D，训练生成器G
        optimizerG.zero_grad()
        # 让D尽可能把G生成的假图判别为1
        label.data.fill_(real_label)
        label = label.to(device)
        output = netD(fake)
        errG = criterion(output, label)
        errG.backward()
        optimizerG.step()

        print('[%d/%d][%d/%d] Loss_D: %.3f Loss_G %.3f'
              % (epoch, opt.epoch, i, len(dataloader), errD.item(), errG.item()))

    vutils.save_image(fake.data,
                      '%s/fake_samples_epoch_%03d.png' % (opt.outf, epoch),
                      normalize=True)
torch.save(netG.state_dict(), '%s/netG_%03d.pth' % (opt.outf, epoch))
torch.save(netD.state_dict(), '%s/netD_%03d.pth' % (opt.outf, epoch))




