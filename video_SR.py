import time
import cv2
import imutils
import numpy as np
from tqdm import tqdm
from Network import Generator
from utils import *


large_kernel_size = 9
small_kernel_size = 3
n_channels = 64
n_blocks = 16
scaling_factor = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    model = "./model/generator.pth"

    checkpoint = torch.load(model, map_location=device)
    generator = Generator(large_kernel_size=large_kernel_size,
                        small_kernel_size=small_kernel_size,
                        n_channels=n_channels,
                        n_blocks=n_blocks,
                        scaling_factor=scaling_factor)

    generator = generator.to(device)
    generator.load_state_dict(checkpoint['generator'])

    generator.eval()
    model = generator

    video_path = "./results/test3.mp4"
    vs = cv2.VideoCapture(video_path)
    (W, H) = (None, None)
    frameIndex = 0

    try:
        prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
            else cv2.CAP_PROP_FRAME_COUNT
        total = int(vs.get(prop))
        print("Total number of frames {}".format(total))

    except:
        print("could not determine # of frames in video")
        print("no approx. completion time can be provided")
        total = -1

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    ret, frame = vs.read()
    vw = frame.shape[1] * scaling_factor
    vh = frame.shape[0] * scaling_factor
    print("Video size ：{} * {}".format(vw, vh))
    output_video = cv2.VideoWriter(video_path.replace(".mp4", "-SR.avi"), fourcc, 20.0, (vw, vh))
    output_video_orig = cv2.VideoWriter(video_path.replace(".mp4", "-ORG.avi"),
                                        fourcc, 20.0, (vw, vh))

    for fr in tqdm(range(total)):
        (grabbed, frame) = vs.read()

        if not grabbed:
            break

        if W is None or H is None:
            (H, W) = frame.shape[:2]

        img = Image.fromarray(frame)
        img = img.convert('RGB')

        Bicubic_img = Image.fromarray(frame)
        Bicubic_img = Bicubic_img.convert('RGB')
        Bicubic_img = Bicubic_img.resize((int(Bicubic_img.width * scaling_factor), int(Bicubic_img.height * scaling_factor)), Image.BICUBIC)

        lr_img = convert_image(img, source='pil', target='imagenet-norm')
        lr_img.unsqueeze_(0)

        start = time.time()

        lr_img = lr_img.to(device)  # (1, 3, w, h ), imagenet-normed

        with torch.no_grad():
            sr_img = model(lr_img).squeeze(0).cpu().detach()  # (1, 3, w*scale, h*scale), in [-1, 1]
            sr_img = convert_image(sr_img, source='[-1, 1]', target='pil')

        sr_img = np.array(sr_img)
        Bicubic_img = np.array(Bicubic_img)

        cv2.imshow('Stream', sr_img)
        output_video.write(sr_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frameIndex += 1

        output_video_orig.write(Bicubic_img)
        if frameIndex >= total:
            print("Finished")
            output_video.release()
            output_video_orig.release()
            vs.release()
            exit()
