import os 
import numpy as np 
import cv2 
from PIL import Image
import h5py
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-in', '--input_path', type=str, default='./test')
    parser.add_argument('-out', '--out_path', type=str, default='./submission')
    parser.add_argument('-t', '--thresd', type=float, default=0.65)
    parser.add_argument('-a', '--ave', action='store_true', default=False)
    args = parser.parse_args()
    print(args)

    interp = cv2.INTER_LINEAR
    # interp = cv2.INTER_NEAREST

    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)
    
    thresd = args.thresd
    ave = args.ave
    w_x0_f = 0.25
    w_x2_f = 0.15
    w_x4_f = 0.1
    w_x0_t = 0.1
    w_x2_t = 0.25
    w_x4_t = 0.15
    img_list = os.listdir(args.input_path)
    t_x4_0 = './caches/twonet_x4_0'
    t_x4_1 = './caches/twonet_x4_1'
    t_x2_0 = './caches/twonet_x2_0'
    t_x2_1 = './caches/twonet_x2_1'
    t_x0_0 = './caches/twonet_x0_0'
    t_x0_1 = './caches/twonet_x0_1'
    f_x4_0 = './caches/fusionnet_x4_0'
    f_x4_1 = './caches/fusionnet_x4_1'
    f_x2_0 = './caches/fusionnet_x2_0'
    f_x2_1 = './caches/fusionnet_x2_1'
    f_x0_0 = './caches/fusionnet_x0_0'
    f_x0_1 = './caches/fusionnet_x0_1'

    for f_img in img_list:
        # Read rtesults
        # name = f_img[:-4] + '.hdf'
        # with h5py.File(os.path.join(t_x4, name), 'r') as f:
        #     x4_img_t = f['main'][:]
        # with h5py.File(os.path.join(t_x2, name), 'r') as f:
        #     x2_img_t = f['main'][:]
        # with h5py.File(os.path.join(t_x0, name), 'r') as f:
        #     x0_img_t = f['main'][:]
        # with h5py.File(os.path.join(f_x4, name), 'r') as f:
        #     x4_img_f = f['main'][:]
        # with h5py.File(os.path.join(f_x2, name), 'r') as f:
        #     x2_img_f = f['main'][:]
        # with h5py.File(os.path.join(f_x0, name), 'r') as f:
        #     x0_img_f = f['main'][:]
        name = f_img[:-4] + '.tiff'
        x4_img_t_0 = cv2.imread(os.path.join(t_x4_0, name), cv2.IMREAD_GRAYSCALE)
        x4_img_t_1 = cv2.imread(os.path.join(t_x4_1, name), cv2.IMREAD_GRAYSCALE)
        x2_img_t_0 = cv2.imread(os.path.join(t_x2_0, name), cv2.IMREAD_GRAYSCALE)
        x2_img_t_1 = cv2.imread(os.path.join(t_x2_1, name), cv2.IMREAD_GRAYSCALE)
        x0_img_t_0 = cv2.imread(os.path.join(t_x0_0, name), cv2.IMREAD_GRAYSCALE)
        x0_img_t_1 = cv2.imread(os.path.join(t_x0_1, name), cv2.IMREAD_GRAYSCALE)

        x4_img_f_0 = cv2.imread(os.path.join(f_x4_0, name), cv2.IMREAD_GRAYSCALE)
        x4_img_f_1 = cv2.imread(os.path.join(f_x4_1, name), cv2.IMREAD_GRAYSCALE)
        x2_img_f_0 = cv2.imread(os.path.join(f_x2_0, name), cv2.IMREAD_GRAYSCALE)
        x2_img_f_1 = cv2.imread(os.path.join(f_x2_1, name), cv2.IMREAD_GRAYSCALE)
        x0_img_f_0 = cv2.imread(os.path.join(f_x0_0, name), cv2.IMREAD_GRAYSCALE)
        x0_img_f_1 = cv2.imread(os.path.join(f_x0_1, name), cv2.IMREAD_GRAYSCALE)


        # up-sampling
        x4_img_t_0 = cv2.resize(x4_img_t_0, (0,0), fx=4, fy=4, interpolation=interp)
        x4_img_t_1 = cv2.resize(x4_img_t_1, (0,0), fx=4, fy=4, interpolation=interp)
        x4_img_t_0 = x4_img_t_0.astype(np.float32) / 255.0
        x4_img_t_1 = x4_img_t_1.astype(np.float32) / 255.0
        x4_img_t = (x4_img_t_0 + x4_img_t_1) / 2

        x2_img_t_0 = cv2.resize(x2_img_t_0, (0,0), fx=2, fy=2, interpolation=interp)
        x2_img_t_1 = cv2.resize(x2_img_t_1, (0,0), fx=2, fy=2, interpolation=interp)
        x2_img_t_0 = x2_img_t_0.astype(np.float32) / 255.0
        x2_img_t_1 = x2_img_t_1.astype(np.float32) / 255.0
        x2_img_t = (x2_img_t_0 + x2_img_t_1) / 2

        x4_img_f_0 = cv2.resize(x4_img_f_0, (0,0), fx=4, fy=4, interpolation=interp)
        x4_img_f_1 = cv2.resize(x4_img_f_1, (0,0), fx=4, fy=4, interpolation=interp)
        x4_img_f_0 = x4_img_f_0.astype(np.float32) / 255.0
        x4_img_f_1 = x4_img_f_1.astype(np.float32) / 255.0
        x4_img_f = (x4_img_f_0 + x4_img_f_1) / 2

        x2_img_f_0 = cv2.resize(x2_img_f_0, (0,0), fx=2, fy=2, interpolation=interp)
        x2_img_f_1 = cv2.resize(x2_img_f_1, (0,0), fx=2, fy=2, interpolation=interp)
        x2_img_f_0 = x2_img_f_0.astype(np.float32) / 255.0
        x2_img_f_1 = x2_img_f_1.astype(np.float32) / 255.0
        x2_img_f = (x2_img_f_0 + x2_img_f_1) / 2

        x0_img_t_0 = x0_img_t_0.astype(np.float32) / 255.0
        x0_img_t_1 = x0_img_t_1.astype(np.float32) / 255.0
        x0_img_f_0 = x0_img_f_0.astype(np.float32) / 255.0
        x0_img_f_1 = x0_img_f_1.astype(np.float32) / 255.0
        x0_img_t = (x0_img_t_0 + x0_img_t_1) / 2
        x0_img_f = (x0_img_f_0 + x0_img_f_1) / 2

        # crop
        x4_img_t = x4_img_t[141:141+9959, 141:141+9958]
        x2_img_t = x2_img_t[141:141+9959, 141:141+9958]
        x0_img_t = x0_img_t[141:141+9959, 141:141+9958]

        x4_img_f = x4_img_f[141:141+9959, 141:141+9958]
        x2_img_f = x2_img_f[141:141+9959, 141:141+9958]
        x0_img_f = x0_img_f[141:141+9959, 141:141+9958]

        if ave:
            # fusion = (x0_img + x2_img + x4_img) / 3.0
            fusion = (x0_img_f+x2_img_f+x4_img_f+x0_img_t+x2_img_t+x4_img_t) / 6.0
        else:
            fusion = x0_img_f*w_x0_f + x2_img_f*w_x2_f + x4_img_f*w_x4_f + x0_img_t * w_x0_t + x2_img_t*w_x2_t + x4_img_t*w_x4_t
        fusion[fusion<=thresd] = 0
        fusion[fusion>thresd] = 1
        fusion = (fusion * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(args.out_path, f_img[:-4]+'.tiff'), fusion)
    print('Done')
