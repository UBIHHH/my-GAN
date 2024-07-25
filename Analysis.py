import cv2
import numpy as np
import matplotlib.pyplot as plt


lr = cv2.imread('results/test2.png', cv2.IMREAD_GRAYSCALE)
hr = cv2.imread('test_srgan.jpg', cv2.IMREAD_GRAYSCALE)

f = np.fft.fft2(lr)
fshift = np.fft.fftshift(f)
lr_fft = 20 * np.log(np.abs(fshift))

f = np.fft.fft2(hr)
fshift = np.fft.fftshift(f)
hr_fft = 20 * np.log(np.abs(fshift))

fig, axs = plt.subplots(2, 2, figsize=(12, 12))

axs[0, 0].imshow(lr)
axs[0, 0].set_title('Low Resolution Image')

axs[0, 1].imshow(hr)
axs[0, 1].set_title('High Resolution Image')

axs[1, 0].imshow(lr_fft, cmap='gray')
axs[1, 0].set_title('FFT of Low Resolution Image')

axs[1, 1].imshow(hr_fft, cmap='gray')
axs[1, 1].set_title('FFT of High Resolution Image')

for ax in axs.flat:
    ax.axis('off')

plt.show()