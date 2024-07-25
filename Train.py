
import torch.backends.cudnn as cudnn
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

from Load_Data import LoadData
from Network import Generator, Discriminator
from VGG import TruncatedVGG19
from utils import *

# 数据集参数
data_folder = './data/'
crop_size = 96
scaling_factor = 4

large_kernel_size_g = 9
small_kernel_size_g = 3
n_channels_g = 64
n_blocks_g = 16
srresnet_checkpoint = "./models/checkpoint_srresnet.pth"  # 预训练的SRResNet模型，用来初始化

kernel_size_d = 3
n_channels_d = 64
n_blocks_d = 8
fc_size_d = 1024


batch_size = 64
start_epoch = 1
epochs = 50
workers = 1  # 加载数据线程数量
vgg19_i = 5
vgg19_j = 4
beta = 1e-3
lr = 1e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True
writer = SummaryWriter()


def main():
    global checkpoint, start_epoch, writer

    generator = Generator(large_kernel_size=large_kernel_size_g,
                          small_kernel_size=small_kernel_size_g,
                          n_channels=n_channels_g,
                          n_blocks=n_blocks_g,
                          scaling_factor=scaling_factor)

    discriminator = Discriminator(kernel_size=kernel_size_d,
                                  n_channels=n_channels_d,
                                  n_blocks=n_blocks_d,
                                  fc_size=fc_size_d)

    optimizer_g = torch.optim.Adam(params=filter(lambda p: p.requires_grad, generator.parameters()), lr=lr)
    optimizer_d = torch.optim.Adam(params=filter(lambda p: p.requires_grad, discriminator.parameters()), lr=lr)

    truncated_vgg19 = TruncatedVGG19(i=vgg19_i, j=vgg19_j)
    truncated_vgg19.eval()

    content_loss_criterion = nn.MSELoss()
    adversarial_loss_criterion = nn.BCEWithLogitsLoss()

    generator = generator.to(device)
    discriminator = discriminator.to(device)
    truncated_vgg19 = truncated_vgg19.to(device)
    content_loss_criterion = content_loss_criterion.to(device)
    adversarial_loss_criterion = adversarial_loss_criterion.to(device)
    # 定制化的dataloaders
    train_dataset = LoadData(data_folder, split='train',
                              crop_size=crop_size,
                              scaling_factor=scaling_factor,
                              lr_img_type='imagenet-norm',
                              hr_img_type='imagenet-norm')
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=workers,
                                               pin_memory=True)

    # 开始逐轮训练
    for epoch in range(start_epoch, epochs + 1):

        if epoch == int(epochs / 2):
            for param_group in optimizer_g.param_groups:
                param_group['lr'] /= 10
            for param_group in optimizer_d.param_groups:
                param_group['lr'] /= 10

        generator.train()
        discriminator.train()

        losses_c = AverageMeter()
        losses_a = AverageMeter()
        losses_d = AverageMeter()

        n_iter = len(train_loader)

        for i, (lr_imgs, hr_imgs) in enumerate(train_loader):

            lr_imgs = lr_imgs.to(device)  # (batch_size (N), 3, 24, 24),  imagenet-normed 格式
            hr_imgs = hr_imgs.to(device)  # (batch_size (N), 3, 96, 96),  imagenet-normed 格式

            sr_imgs = generator(lr_imgs)
            sr_imgs = convert_image(
                sr_imgs, source='[-1, 1]',
                target='imagenet-norm')  # (N, 3, 96, 96), imagenet-normed

            sr_imgs_in_vgg_space = truncated_vgg19(sr_imgs)  # batchsize X 512 X 6 X 6
            hr_imgs_in_vgg_space = truncated_vgg19(hr_imgs).detach()  # batchsize X 512 X 6 X 6

            content_loss = content_loss_criterion(sr_imgs_in_vgg_space, hr_imgs_in_vgg_space)

            sr_discriminated = discriminator(sr_imgs)  # (batch X 1)
            adversarial_loss = adversarial_loss_criterion(
                sr_discriminated, torch.ones_like(sr_discriminated))  # 生成器希望生成的图像能够完全迷惑判别器，因此它的预期所有图片真值为1

            perceptual_loss = content_loss + beta * adversarial_loss

            optimizer_g.zero_grad()
            perceptual_loss.backward()

            optimizer_g.step()

            losses_c.update(content_loss.item(), lr_imgs.size(0))
            losses_a.update(adversarial_loss.item(), lr_imgs.size(0))

            hr_discriminated = discriminator(hr_imgs)
            sr_discriminated = discriminator(sr_imgs.detach())

            adversarial_loss = adversarial_loss_criterion(sr_discriminated, torch.zeros_like(sr_discriminated)) + \
                               adversarial_loss_criterion(hr_discriminated, torch.ones_like(
                                   hr_discriminated))

            optimizer_d.zero_grad()
            adversarial_loss.backward()

            optimizer_d.step()

            losses_d.update(adversarial_loss.item(), hr_imgs.size(0))

            if i == (n_iter - 2):
                writer.add_image('SRGAN/epoch_' + str(epoch) + '_1',
                                 make_grid(lr_imgs[:4, :3, :, :].cpu(), nrow=4, normalize=True), epoch)
                writer.add_image('SRGAN/epoch_' + str(epoch) + '_2',
                                 make_grid(sr_imgs[:4, :3, :, :].cpu(), nrow=4, normalize=True), epoch)
                writer.add_image('SRGAN/epoch_' + str(epoch) + '_3',
                                 make_grid(hr_imgs[:4, :3, :, :].cpu(), nrow=4, normalize=True), epoch)

            print('Epoch [{0}/{1}], Iter [{2}/{3}], Content_Loss: {4:.4f}, Adversarial_Loss: {5:.4f}, Discriminator_Loss: {6:.4f}'
                  .format(epoch, epochs, i + 1, n_iter, content_loss.item(), adversarial_loss.item(), adversarial_loss.item()))

        # 手动释放内存
        del lr_imgs, hr_imgs, sr_imgs, hr_imgs_in_vgg_space, sr_imgs_in_vgg_space, hr_discriminated, sr_discriminated  # 手工清除掉缓存

        writer.add_scalar('SRGAN/Loss_c', losses_c.val, epoch)
        writer.add_scalar('SRGAN/Loss_a', losses_a.val, epoch)
        writer.add_scalar('SRGAN/Loss_d', losses_d.val, epoch)

        torch.save({
            'epoch': epoch,
            'generator': generator.state_dict(),
            'discriminator': discriminator.state_dict(),
            'optimizer_g': optimizer_g.state_dict(),
            'optimizer_g': optimizer_g.state_dict(),
        }, 'results/checkpoint_srgan.pth')
    torch.save(generator.state_dict(), 'results/generator.pth')
    # 训练结束关闭监控
    writer.close()


if __name__ == '__main__':
    main()
