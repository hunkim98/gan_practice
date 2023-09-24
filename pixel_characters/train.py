import numpy as np
import torch
from torch import nn
from model import Generator, Discriminator
from dataloader import PixelCharacterDataset, DataLoader
from matplotlib import pyplot as plt
from Diff_Augment import DiffAugment

torch.manual_seed(112)

def init_weights(m):
    if type(m) == nn.ConvTranspose2d:
        nn.init.normal_(m.weight, 0.0, 0.02)
    elif type(m) == nn.BatchNorm2d:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.constant_(m.bias, 0)
        
def plot_images(imgs, grid_size = 10, epoch = 0, loss = 0):
    """
    imgs: vector containing all the numpy images
    grid_size: 2x2 or 5x5 grid containing images
    """
     
    fig = plt.figure(figsize = (8, 8))
    columns = rows = grid_size
    plt.title(f"Training Images at epoch {epoch}, loss: {loss}")

    for i in range(1, columns*rows +1):
        if i >= len(imgs):
            break
        plt.axis("off")
        fig.add_subplot(rows, columns, i)
        plt.imshow(imgs[i])
    plt.savefig(f"epoch/training_images_epoch:{epoch}_loss:{loss}.png")
    plt.show()
    plt.close()

device = ""
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(device.type)
    print(device.index)
else:
    device = torch.device("cpu")


generator = Generator().to(device)
discriminator = Discriminator().to(device)
generator.apply(init_weights)
discriminator.apply(init_weights)

imgs = np.load('8bit_characters_50x50.npz')
transpose_imgs = np.transpose(
    np.float32(imgs['arr_0']), (0, 3, 1, 2)
)
dset = PixelCharacterDataset(transpose_imgs)

batch_size = 32

dataloader = DataLoader(dset, batch_size=batch_size, shuffle=True)

lr = 0.001 # learning rate
num_epochs = 1000 # number of epochs
def backward_hook(grad):
    print(grad)
loss_function = nn.BCELoss() # Binary Cross Entropy Loss (since we are doing binary classification - whether the data is real or fake)

# We will use Adam optimizer
optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_generator = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))

diff_policy='color,cutout'


# return

loss_threshold = 2
# going over the entire dataset 10 times
for e in range(num_epochs):
    # pick each batch b of input images: shape of each batch is (32, 3, 50, 50)
    for i, b in enumerate(dataloader):

        # b.shape[0] is the batch size
        # there is a possibility where the last batch in an epoch has a different batch size than the other batches
        # this is because the total number of images in the dataset is not a multiple of the batch size
        if b.shape[0] != batch_size:
            continue

        ##########################
        ## Update Discriminator ##
        ##########################
 
        # Loss on real images
         
        # clear the gradient
        optimizer_discriminator.zero_grad() # set the gradients to 0 at start of each loop because gradients are accumulated on subsequent backward passes
        # compute the D model output
        b_t = DiffAugment(b, policy=diff_policy)
        yhat = discriminator(b_t.to(device)).view(-1) # view(-1) reshapes a 4-d tensor of shape (2,1,1,1) to 1-d tensor with 2 values only
        # specify target labels or true labels
        target = torch.ones(len(b_t), dtype=torch.float, device=device)
        # calculate loss
        loss_real = loss_function(yhat, target)
        # calculate gradients -  or rather accumulation of gradients on loss tensor
        loss_real.backward()
 
        # Loss on fake images
 
        # generate batch of fake images using G
        # Step1: creating noise to be fed as input to G
        noise = torch.randn(len(b_t), 100, 1, 1, device = device)
        # Step 2: feed noise to G to create a fake img (this will be reused when updating G)
        fake_img = generator(noise) 
        fake_img = DiffAugment(fake_img, policy=diff_policy)
 
        # compute D model output on fake images
        yhat = discriminator(fake_img.detach()).view(-1).to(device) # .cuda() is essential because our input i.e. fake_img is on gpu but model isnt (runtimeError thrown); detach is imp: Basically, only track steps on your generator optimizer when training the generator, NOT the discriminator. 
        # specify target labels
        target = torch.zeros(len(b_t), dtype=torch.float, device=device)
        # calculate loss
        loss_fake = loss_function(yhat, target)
        # calculate gradients
        loss_fake.backward()
 
        # total error on D
        loss_disc = loss_real + loss_fake
 
        # Update weights of D
        optimizer_discriminator.step()
 
        ##########################
        #### Update Generator ####
        ##########################
 
        # clear gradient
        optimizer_generator.zero_grad()
        # pass fake image through D
        yhat = discriminator(fake_img).view(-1).to(device)
        # specify target variables - remember G wants D *to think* these are real images so label is 1
        target = torch.ones(len(b), dtype=torch.float, device=device)
        # calculate loss
        loss_gen = loss_function(yhat, target)
        # calculate gradients
        loss_gen.backward()
        # update weights on G
        optimizer_generator.step()
 
 
        ####################################
        #### Plot some Generator images ####
        ####################################
 
        if loss_gen < loss_threshold:
            torch.save(generator.state_dict(), f"generator_epoch:{e}.pth")
            torch.save(discriminator.state_dict(), f"discriminator_epoch:{e}.pth")
            loss_threshold = loss_gen

        # during every epoch, print images at every 10th iteration.
        if e % 10 == 0 and i == len(dataloader)-3:
            # convert the fake images from (b_size, 3, 32, 32) to (b_size, 32, 32, 3) for plotting 
            generated = generator(torch.randn(batch_size, 100, 1, 1, device = device))
            img_plot = np.transpose(generated.detach().cpu(), (0,2,3,1))
            img_plot = (img_plot + 1)/2
            print("********************")
            print(" Epoch %d and iteration %d " % (e, i))
            print("Discriminator loss: %f" % (loss_disc))
            print("Generator loss: %f" % (loss_gen))
            plot_images(img_plot, grid_size=5, epoch=e, loss=loss_gen)


# torch.save(generator.state_dict(), "generator.pth")
# torch.save(discriminator.state_dict(), "discriminator.pth")
