import glob
import os
import cv2
import numpy as np

image_size = 256

x_split = [[],[]]
for i, imgdir in enumerate(['/home/storage/wuxian/ATR/train/image/*','/home/storage/wuxian/ATR/test/image/*']):
    paths = glob.glob(imgdir)
    for path in paths:
        img = cv2.imread(path)
        img = cv2.resize(img, (image_size, image_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x_split[i].append(img)

    x_split[i] = np.array(x_split[i], dtype=np.uint8)

if not os.path.exists('./npy'):
    os.mkdir('./npy')
print (x_split[0].shape, x_split[1].shape)
np.save('./npy/x_train.npy', x_split[0])
np.save('./npy/x_test.npy', x_split[1])

