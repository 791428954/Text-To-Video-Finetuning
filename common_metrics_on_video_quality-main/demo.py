import torch
from calculate_fvd import calculate_fvd
from calculate_psnr import calculate_psnr
from calculate_ssim import calculate_ssim
# from calculate_lpips import calculate_lpips

# ps: pixel value should be in [0, 1]!
import numpy as np
import cv2
 
cap = cv2.VideoCapture('XXX.mp4')
wid = int(cap.get(3))
hei = int(cap.get(4))
framerate = int(cap.get(5))
framenum = int(cap.get(7))
 
video = np.zeros((framenum,hei,wid,3),dtype='float16')
cnt = 0
while(cap.isOpened()):
    a,b=cap.read()
    cv2.imshow('%d'%cnt, b)
    cv2.waitKey(20)
    b = b.astype('float16')/255
    video[cnt]=b
    print(cnt)


NUMBER_OF_VIDEOS = 8
VIDEO_LENGTH = 30
CHANNEL = 3
SIZE = 64
videos1 = torch.zeros(NUMBER_OF_VIDEOS, VIDEO_LENGTH, CHANNEL, SIZE, SIZE, requires_grad=False)
videos2 = torch.ones(NUMBER_OF_VIDEOS, VIDEO_LENGTH, CHANNEL, SIZE, SIZE, requires_grad=False)
device = torch.device("cuda")
# device = torch.device("cpu")

import json
result = {}
result['fvd'] = calculate_fvd(videos1, videos2, device)
result['ssim'] = calculate_ssim(videos1, videos2)
result['psnr'] = calculate_psnr(videos1, videos2)
# result['lpips'] = calculate_lpips(videos1, videos2, device)
print(json.dumps(result, indent=4))
