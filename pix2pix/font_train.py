import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from font_gan.model import Generator, Discriminator

from PIL import Image
import matplotlib.pyplot as plt
from math import log10 # For metric function

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


# split concatted image into two images
def split_image(image):
    width, height = image.size
    width = int(width / 2)
    left = image.crop((0, 0, width, height))
    right = image.crop((width, 0, width * 2, height))
    return left, right

# Load Dataset from ImageFolder
class Dataset(data.Dataset): # torch기본 Dataset 상속받기
    def __init__(self, image_dir, direction):
        super(Dataset, self).__init__() # 초기화 상속
        self.direction = direction # 
        self.a_path = os.path.join(image_dir, "a") # a는 건물 사진
        self.b_path = os.path.join(image_dir, "b") # b는 Segmentation Mask
        self.image_filenames = [x for x in os.listdir(self.a_path)] # a 폴더에 있는 파일 목록
        self.transform = transforms.Compose([transforms.Resize((256, 256)), # 이미지 크기 조정
                                             transforms.Grayscale(),
                                            transforms.ToTensor(), # Numpy -> Tensor
                                             transforms.Normalize(mean=(0.5,), 
                                                std=(0.5,)) # Normalization : -1 ~ 1 range
                                            ])
        self.len = len(self.image_filenames)
    
    def __getitem__(self, index):
        
        # 건물사진과 Segmentation mask를 각각 a,b 폴더에서 불러오기
        a = Image.open(os.path.join(self.a_path, self.image_filenames[index])).convert('L') # 건물 사진
        b = Image.open(os.path.join(self.b_path, self.image_filenames[index])).convert('L') # Segmentation 사진
        
        # 이미지 전처리
        a = self.transform(a)
        b = self.transform(b)
        
        if self.direction == "a2b": # 건물 -> Segmentation
            return a, b
        else:  # Segmentation -> 건물
            return b, a
    
    def __len__(self):
        return self.len
        
train_dataset = Dataset("./data/fonts/train/", "b2a")
test_dataset = Dataset("./data/fonts/test/", "b2a")

train_loader = DataLoader(dataset=train_dataset, num_workers=0, batch_size=32, shuffle=True) # Shuffle
test_loader = DataLoader(dataset=test_dataset, num_workers=0, batch_size=32, shuffle=False)

# -1 ~ 1사이의 값을 0~1사이로 만들어준다
def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

# 이미지 시각화 함수
def show_images(real_a, real_b, fake_b):
    plt.figure(figsize=(30,90))
    plt.subplot(131)
    plt.imshow(real_a.cpu().data.numpy().transpose(1,2,0))
    plt.xticks([])
    plt.yticks([])
    
    plt.subplot(132)
    plt.imshow(real_b.cpu().data.numpy().transpose(1,2,0))
    plt.xticks([])
    plt.yticks([])
    
    plt.subplot(133)
    plt.imshow(fake_b.cpu().data.numpy().transpose(1,2,0))
    plt.xticks([])
    plt.yticks([])
    
    plt.show()


# Generator와 Discriminator를 GPU로 보내기
G = Generator().cuda()
D = Discriminator().cuda()

criterionL1 = nn.L1Loss().cuda()
criterionMSE = nn.MSELoss().cuda()

# Setup optimizer
g_optimizer = optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
d_optimizer = optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))

epoch_amount = 2

# Train
for epoch in range(1, epoch_amount):
    for i, (real_a, real_b) in enumerate(train_loader, 1):
        # forward
        real_a, real_b = real_a.cuda(), real_b.cuda()
        real_label = torch.ones(1).cuda()
        fake_label = torch.zeros(1).cuda()
        
        fake_b = G(real_a) # G가 생성한 fake Segmentation mask
        
        #============= Train the discriminator =============#
        # train with fake
        fake_ab = torch.cat((real_a, fake_b), 1)
        pred_fake = D.forward(fake_ab.detach())
        loss_d_fake = criterionMSE(pred_fake, fake_label)

        # train with real
        real_ab = torch.cat((real_a, real_b), 1)
        pred_real = D.forward(real_ab)
        loss_d_real = criterionMSE(pred_real, real_label)
        
        # Combined D loss
        loss_d = (loss_d_fake + loss_d_real) * 0.5
        
        # Backprop + Optimize
        D.zero_grad()
        loss_d.backward()
        d_optimizer.step()

        #=============== Train the generator ===============#
        # First, G(A) should fake the discriminator
        fake_ab = torch.cat((real_a, fake_b), 1)
        pred_fake = D.forward(fake_ab)
        loss_g_gan = criterionMSE(pred_fake, real_label)

        # Second, G(A) = B
        loss_g_l1 = criterionL1(fake_b, real_b) * 10
        
        loss_g = loss_g_gan + loss_g_l1
        
        # Backprop + Optimize
        G.zero_grad()
        D.zero_grad()
        loss_g.backward()
        g_optimizer.step()
        
        if i % 200 == 0:
            print('======================================================================================================')
            print('Epoch [%d/%d], Step[%d/%d], d_loss: %.4f, g_loss: %.4f'
                  % (epoch, epoch_amount, i, len(train_loader), loss_d.item(), loss_g.item()))
            print('======================================================================================================')
            # show_images(denorm(real_a.squeeze()), denorm(real_b.squeeze()), denorm(fake_b.squeeze()))
    
    # Save the models
    torch.save(G.state_dict(), './min_generator_bw.pkl')