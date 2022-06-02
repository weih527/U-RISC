import os
import cv2
import argparse
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-in', '--input_path', type=str, default='./test')
    parser.add_argument('-x0', '--out_x0', type=str, default='./test_x0')
    parser.add_argument('-x2', '--out_x2', type=str, default='./test_x2')
    parser.add_argument('-x4', '--out_x4', type=str, default='./test_x4')
    args = parser.parse_args()
    print(args)

    out_path = [args.out_x0, args.out_x2, args.out_x4]
    for path in out_path:
        if not os.path.exists(path):
            os.makedirs(path)
    
    img_list = os.listdir(args.input_path)
    for f_img in img_list:
        if f_img[-4:] == '.png':
            raw = cv2.imread(os.path.join(args.input_path, f_img))
            raw_ = np.zeros((10240,10240,3), dtype=np.uint8)
            raw_[141:141+9959, 141:141+9958, :] = raw
            raw = raw_
            del raw_
            cv2.imwrite(os.path.join(args.out_x0, f_img), raw)
            raw2 = cv2.resize(raw, (0,0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(os.path.join(args.out_x2, f_img), raw2)
            raw4 = cv2.resize(raw, (0,0), fx=0.25, fy=0.25, interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(os.path.join(args.out_x4, f_img), raw4)
    print('Done: image processing!')