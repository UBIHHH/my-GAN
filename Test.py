from utils import *
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from Load_Data import LoadData
from Network import Generator
import time

large_kernel_size = 9
small_kernel_size = 3
n_channels = 64
n_blocks = 16
scaling_factor = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':

    data_folder = "./data/"
    test_data_names = ["Set14"]

    srgan_checkpoint = "./model/generator.pth"

    checkpoint = torch.load(srgan_checkpoint)
    generator = Generator(large_kernel_size=large_kernel_size,
                          small_kernel_size=small_kernel_size,
                          n_channels=n_channels,
                          n_blocks=n_blocks,
                          scaling_factor=scaling_factor)
    generator = generator.to(device)
    generator.load_state_dict(checkpoint['generator'])

    generator.eval()
    model = generator

    for test_data_name in test_data_names:
        test_dataset = LoadData(data_folder,
                                 split='test',
                                 crop_size=0,
                                 scaling_factor=4,
                                 lr_img_type='imagenet-norm',
                                 hr_img_type='[-1, 1]',
                                 test_data_name=test_data_name)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1,
                                                  pin_memory=True)

        PSNRs = AverageMeter()
        SSIMs = AverageMeter()

        start = time.time()

        with torch.no_grad():
            for i, (lr_imgs, hr_imgs) in enumerate(test_loader):
                lr_imgs = lr_imgs.to(device)  # (batch_size (1), 3, w / 4, h / 4), imagenet-normed
                hr_imgs = hr_imgs.to(device)  # (batch_size (1), 3, w, h), in [-1, 1]

                sr_imgs = model(lr_imgs)  # (1, 3, w, h), in [-1, 1]

                # PSNR and SSIM
                sr_imgs_y = convert_image(sr_imgs, source='[-1, 1]', target='y-channel').squeeze(
                    0)  # (w, h), in y-channel
                hr_imgs_y = convert_image(hr_imgs, source='[-1, 1]', target='y-channel').squeeze(
                    0)  # (w, h), in y-channel
                psnr = peak_signal_noise_ratio(hr_imgs_y.cpu().numpy(), sr_imgs_y.cpu().numpy(),
                                               data_range=255.)
                ssim = structural_similarity(hr_imgs_y.cpu().numpy(), sr_imgs_y.cpu().numpy(),
                                             data_range=255.)
                PSNRs.update(psnr, lr_imgs.size(0))
                SSIMs.update(ssim, lr_imgs.size(0))

        print('PSNR  {psnrs.avg:.3f}'.format(psnrs=PSNRs))
        print('SSIM  {ssims.avg:.3f}'.format(ssims=SSIMs))
        print('avg time for a image is {:.3f} sec'.format((time.time() - start) / len(test_dataset)))

    print("\n")
