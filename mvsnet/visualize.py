import numpy as np
import cv2
import argparse
import matplotlib.pyplot as plt
from preprocess import load_pfm
from preprocess import get_normal_from_depth
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('depth_path')
    args = parser.parse_args()
    path = args.depth_path
    depth_files = os.listdir(os.path.join(path, 'depth'))
    mask_files = os.listdir(os.path.join(path, 'ground_truth'))
    depth_init_files = os.listdir(os.path.join(path, 'init'))
    prob_files = os.listdir(os.path.join(path, 'prob'))
    for i in range(0, len(depth_files)):
        if depth_files[i].endswith('.pfm'):
            depth_image = load_pfm(open(os.path.join(path, 'depth', depth_files[i]), 'rb'))
            #normal = get_normal_from_depth(depth_image)
            gt_image = load_pfm(open(os.path.join(path, 'ground_truth', mask_files[i]), 'rb'))
            #normal = get_normal_from_depth(gt_image)
            gt_image = cv2.resize(gt_image, (0, 0), None, 0.25, 0.25)
            init_image = load_pfm(open(os.path.join(path, 'init', depth_init_files[i]), 'rb'))
            prob_image = load_pfm(open(os.path.join(path, 'prob', prob_files[i]), 'rb'))
            mask = gt_image < 0
            depth_image[mask] = float('nan')

            init_image[mask] = float('nan')
            name, ext = os.path.splitext(depth_files[i])
            plt.imshow(depth_image, 'rainbow')
            plt.savefig(os.path.join(path, 'depth', name+'.png'))
            plt.close('all')
            normal = get_normal_from_depth(cv2.resize(depth_image, (0, 0), None, fx=2, fy=2))
            normal = 255*(normal+1)/2
            #cv2.imwrite('test.png', normal.astype(np.uint8))
            plt.imshow(normal.astype(np.uint8), 'rainbow')
            plt.savefig(os.path.join(path, 'depth', name + 'normal.png'))
            plt.close('all')

            name, ext = os.path.splitext(depth_init_files[i])
            plt.imshow(init_image, 'rainbow')
            plt.savefig(os.path.join(path, 'init', name+'.png'))
            plt.close('all')
            normal = get_normal_from_depth(cv2.resize(init_image, (0, 0), None, fx=2, fy=2))
            normal = 255 * (normal + 1) / 2
            plt.imshow(normal.astype(np.uint8), 'rainbow')
            plt.savefig(os.path.join(path, 'init', name + 'normal.png'))
            plt.close('all')

            name, ext = os.path.splitext(prob_files[i])
            plt.imshow(prob_image, 'rainbow')
            plt.savefig(os.path.join(path, 'prob', name + '.png'))
            plt.close('all')
            name, ext = os.path.splitext(mask_files[i])
            plt.imshow(gt_image, 'rainbow')
            plt.savefig(os.path.join(path, 'ground_truth', name + '.png'))
            plt.close('all')
            normal = get_normal_from_depth(cv2.resize(gt_image, (0,0), None, fx=2, fy=2))
            normal = 255 * (normal + 1) / 2
            plt.imshow(normal.astype(np.uint8), 'rainbow')
            plt.savefig(os.path.join(path, 'ground_truth', name + 'normal.png'))
            plt.close('all')

#if __name__ == '__main__':
#    parser = argparse.ArgumentParser()
#    parser.add_argument('depth_path')
#    args = parser.parse_args()
#    depth_path = args.depth_path
#    if depth_path.endswith('npy'):
#        depth_image = np.load(depth_path)
#        depth_image = np.squeeze(depth_image)
#        print('value range: ', depth_image.min(), depth_image.max())
#        plt.imshow(depth_image, 'rainbow')
#        plt.show()
#    elif depth_path.endswith('pfm'):
#        depth_image = load_pfm(open(depth_path))
#        ma = np.ma.masked_equal(depth_image, 0.0, copy=False)
#        print('value range: ', ma.min(), ma.max())
#        plt.imshow(depth_image, 'rainbow')
#        plt.show()
#    else:
#        depth_image = cv2.imread(depth_path)
#        ma = np.ma.masked_equal(depth_image, 0.0, copy=False)
#        print('value range: ', ma.min(), ma.max())
#        plt.imshow(depth_image)
#        plt.show()
