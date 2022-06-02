import os
import cv2
import torch
import h5py
import time
import argparse
import numpy as np
from PIL import Image
from fusionnet.model_instanceBN_sigmoid import FusionNet
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from inference_crop_batch import Crop_image
import pdb

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-in', '--input_path', type=str, default='./test')
    parser.add_argument('-out', '--out_path', type=str, default='./caches')
    parser.add_argument('-mp', '--model_path', type=str, default='./models5')
    parser.add_argument('-f', '--num_inference', type=int, default=1)
    parser.add_argument('-c', '--crop_size', type=int, default=1024)
    parser.add_argument('-s', '--stride', type=int, default=512)
    parser.add_argument('-x', '--resolution', type=int, default=0)
    parser.add_argument('-m', '--model', type=str, default=None)
    parser.add_argument('-n', '--number', type=int, default=0)
    parser.add_argument('-t', '--tta', action='store_false', default=True)
    args = parser.parse_args()
    print(args)

    if args.resolution == 0:
        input_path = './test_x0'
    elif args.resolution == 2:
        input_path = './test_x2'
    elif args.resolution == 4:
        input_path = './test_x4'
    else:
        raise AttributeError('No this resolution!')

    out_path = args.out_path + '/fusionnet_x' + str(args.resolution) + '_' + str(args.number)
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    
    # Load model
    print('Load model...')
    model_path = args.model_path
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = FusionNet(input_nc=3, output_nc=1, ngf=32).to(device)
    cuda_count = torch.cuda.device_count()
    if cuda_count > 1:
        print('%d GPUs ... ' % cuda_count, flush=True)
        model = nn.DataParallel(model)
    else:
        print('a single GPU ... ', flush=True)
    ckpt_path = os.path.join(model_path, args.model+'.ckpt')
    if os.path.isfile(ckpt_path):
        checkpoint = torch.load(ckpt_path)
        if cuda_count > 1:
            model.load_state_dict(checkpoint['model_weights'])
        else:
            new_state_dict = OrderedDict()
            state_dict = checkpoint['model_weights']
            for k, v in state_dict.items():
                name = k[7:] # remove module.
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
    else:
        raise AttributeError('No checkpoint found at %s' % model_path)

    # Inference
    PAD = 0
    img_list = os.listdir(input_path)
    if len(img_list) % 2 != 0:
        print('***the number of inference images is 1!')
        num_inference = 1
    else:
        num_inference = args.num_inference
    if num_inference == 1:
        for f_img in img_list:
            print('Inference: ' + f_img, end=' ')
            start = time.time()
            raw = np.asarray(Image.open(os.path.join(input_path, f_img)).convert('L'))
            raw = raw.astype(np.float32) / 255.0

            results_aug = np.zeros_like(raw, dtype=np.float32)
            raw_aug = []
            raw_flipud = np.flipud(raw)
            if args.tta:
                raw_aug.append(raw)
                raw_aug.append(np.rot90(raw, 1))
                raw_aug.append(np.rot90(raw, 2))
                raw_aug.append(np.rot90(raw, 3))
                # raw_aug.append(raw_flipud)
                # raw_aug.append(np.rot90(raw_flipud, 1))
                # raw_aug.append(np.rot90(raw_flipud, 2))
                # raw_aug.append(np.rot90(raw_flipud, 3))
            else:
                raw_aug.append(raw)
            raw_aug = np.asarray(raw_aug)
            crop_img = Crop_image(raw_aug,crop_size=args.crop_size,overlap=args.stride)
            for i in range(crop_img.num):
                for j in range(crop_img.num):
                    raw_crop = crop_img.gen(i, j)
                    if crop_img.dim == 3:
                        raw_crop_ = raw_crop[:, np.newaxis, :, :].copy()
                    else:
                        raw_crop_ = raw_crop[np.newaxis, np.newaxis, :, :].copy()
                    raw_crop_ = np.concatenate((raw_crop_,raw_crop_,raw_crop_), axis=1)
                    inputs = torch.Tensor(raw_crop_).to(device)
                    inputs = F.pad(inputs, (PAD, PAD, PAD, PAD))
                    with torch.no_grad():
                        pred = model(inputs)
                    pred = F.pad(pred, (-PAD, -PAD, -PAD, -PAD))
                    pred = pred.data.cpu().numpy()
                    pred = np.squeeze(pred)
                    crop_img.save(i, j, pred)
            results = crop_img.result()
            results[results<=0] = 0
            results[results>1] = 1

            inference_results = []
            if args.tta:
                inference_results.append(results[0])
                inference_results.append(np.rot90(results[1], 3))
                inference_results.append(np.rot90(results[2], 2))
                inference_results.append(np.rot90(results[3], 1))
                # inference_results.append(np.flipud(results[4]))
                # inference_results.append(np.flipud(np.rot90(results[5], 3)))
                # inference_results.append(np.flipud(np.rot90(results[6], 2)))
                # inference_results.append(np.flipud(np.rot90(results[7], 1)))
            else:
                inference_results.append(results[0])
            inference_results = np.array(inference_results)

            results_aug = np.sum(inference_results, axis=0) / inference_results.shape[0]
            print('COST TIME: ', (time.time() - start))
            results_aug = (results_aug * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(out_path, f_img[:-3]+'tiff'), results_aug)
    else:
        half_len = len(img_list) // 2
        img_list1 = img_list[:half_len]
        img_list2 = img_list[half_len:]
        assert len(img_list1) == len(img_list2), 'Error'
        for num in range(half_len):
            img1 = img_list1[num]
            img2 = img_list2[num]
            print('Inference: ' + img1 + ' and ' + img2, end=' ')
            start = time.time()
            raw1 = np.asarray(Image.open(os.path.join(input_path, img1)).convert('L'))
            raw2 = np.asarray(Image.open(os.path.join(input_path, img2)).convert('L'))
            raw1 = raw1.astype(np.float32) / 255.0
            raw2 = raw2.astype(np.float32) / 255.0
            results_aug1 = np.zeros_like(raw1, dtype=np.float32)
            results_aug2 = np.zeros_like(raw2, dtype=np.float32)
            raw_aug = []
            raw_flipud1 = np.flipud(raw1)
            raw_flipud2 = np.flipud(raw2)
            if args.tta:
                raw_aug.append(raw1)
                raw_aug.append(np.rot90(raw1, 1))
                raw_aug.append(np.rot90(raw1, 2))
                raw_aug.append(np.rot90(raw1, 3))
                # raw_aug.append(raw_flipud1)
                # raw_aug.append(np.rot90(raw_flipud1, 1))
                # raw_aug.append(np.rot90(raw_flipud1, 2))
                # raw_aug.append(np.rot90(raw_flipud1, 3))
                raw_aug.append(raw2)
                raw_aug.append(np.rot90(raw2, 1))
                raw_aug.append(np.rot90(raw2, 2))
                raw_aug.append(np.rot90(raw2, 3))
                # raw_aug.append(raw_flipud2)
                # raw_aug.append(np.rot90(raw_flipud2, 1))
                # raw_aug.append(np.rot90(raw_flipud2, 2))
                # raw_aug.append(np.rot90(raw_flipud2, 3))
            else:
                raw_aug.append(raw1)
                raw_aug.append(raw2)
            raw_aug = np.array(raw_aug)
            crop_img = Crop_image(raw_aug,crop_size=args.crop_size,overlap=args.stride)
            for i in range(crop_img.num):
                for j in range(crop_img.num):
                    raw_crop = crop_img.gen(i, j)
                    if crop_img.dim == 3:
                        raw_crop_ = raw_crop[:, np.newaxis, :, :].copy()
                    else:
                        raw_crop_ = raw_crop[np.newaxis, np.newaxis, :, :].copy()
                    raw_crop_ = np.concatenate((raw_crop_,raw_crop_,raw_crop_), axis=1)
                    inputs = torch.Tensor(raw_crop_).to(device)
                    inputs = F.pad(inputs, (PAD, PAD, PAD, PAD))
                    with torch.no_grad():
                        pred = model(inputs)
                    pred = F.pad(pred, (-PAD, -PAD, -PAD, -PAD))
                    pred = pred.data.cpu().numpy()
                    pred = np.squeeze(pred)
                    crop_img.save(i, j, pred)
            results = crop_img.result()
            results[results<=0] = 0
            results[results>1] = 1

            inference_results1 = []
            inference_results2 = []
            if args.tta:
                # inference_results1.append(results[0])
                # inference_results1.append(np.rot90(results[1], 3))
                # inference_results1.append(np.rot90(results[2], 2))
                # inference_results1.append(np.rot90(results[3], 1))
                # inference_results1.append(np.flipud(results[4]))
                # inference_results1.append(np.flipud(np.rot90(results[5], 3)))
                # inference_results1.append(np.flipud(np.rot90(results[6], 2)))
                # inference_results1.append(np.flipud(np.rot90(results[7], 1)))
                # inference_results2.append(results[8])
                # inference_results2.append(np.rot90(results[9], 3))
                # inference_results2.append(np.rot90(results[10], 2))
                # inference_results2.append(np.rot90(results[11], 1))
                # inference_results2.append(np.flipud(results[12]))
                # inference_results2.append(np.flipud(np.rot90(results[13], 3)))
                # inference_results2.append(np.flipud(np.rot90(results[14], 2)))
                # inference_results2.append(np.flipud(np.rot90(results[15], 1)))
                inference_results1.append(results[0])
                inference_results1.append(np.rot90(results[1], 3))
                inference_results1.append(np.rot90(results[2], 2))
                inference_results1.append(np.rot90(results[3], 1))
                inference_results2.append(results[4])
                inference_results2.append(np.rot90(results[5], 3))
                inference_results2.append(np.rot90(results[6], 2))
                inference_results2.append(np.rot90(results[7], 1))
            else:
                inference_results1.append(results[0])
                inference_results2.append(results[1])
            inference_results1 = np.array(inference_results1)
            inference_results2 = np.array(inference_results2)

            results_aug1 = np.sum(inference_results1, axis=0) / inference_results1.shape[0]
            results_aug2 = np.sum(inference_results2, axis=0) / inference_results2.shape[0]
            print('COST TIME: ', (time.time() - start))
            results_aug1 = (results_aug1 * 255).astype(np.uint8)
            results_aug2 = (results_aug2 * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(out_path, img1[:-4]+'.tiff'), results_aug1)
            cv2.imwrite(os.path.join(out_path, img2[:-4]+'.tiff'), results_aug2)
    print('***Done***')