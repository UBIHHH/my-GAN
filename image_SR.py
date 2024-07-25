import time
from Network import Generator
from utils import *

imgPath = './results/test2.png'

large_kernel_size = 9
small_kernel_size = 3
n_channels = 64
n_blocks = 16
scaling_factor = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    srgan_checkpoint = "./model/generator.pth"

    checkpoint = torch.load(srgan_checkpoint, map_location=device)
    generator = Generator(large_kernel_size=large_kernel_size,
                          small_kernel_size=small_kernel_size,
                          n_channels=n_channels,
                          n_blocks=n_blocks,
                          scaling_factor=scaling_factor)

    generator = generator.to(device)
    generator.load_state_dict(checkpoint['generator'])

    generator.eval()
    model = generator

    img = Image.open(imgPath, mode='r')
    img = img.convert('RGB')

    Bicubic_img = img.resize((int(img.width * scaling_factor), int(img.height * scaling_factor)), Image.BICUBIC)
    Bicubic_img.save('./results/test_bicubic.jpg')

    lr_img = convert_image(img, source='pil', target='imagenet-norm')
    lr_img.unsqueeze_(0)

    start = time.time()

    lr_img = lr_img.to(device)  # (1, 3, w, h ), imagenet-normed

    with torch.no_grad():
        sr_img = model(lr_img).squeeze(0).cpu().detach()  # (1, 3, w*scale, h*scale), in [-1, 1]
        sr_img = convert_image(sr_img, source='[-1, 1]', target='pil')
        sr_img.save('./results/test_srgan.jpg')

    print('It took {:.3f} seconds'.format(time.time() - start))
