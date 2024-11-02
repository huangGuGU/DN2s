# encoding:utf-8
import os
import torch
from torchvision import utils as vutils
from scipy.io import savemat
from PIL import Image


def save_result_image(save_dir, epoch, fn_list, tensor_img):
    save_dir = os.path.join(save_dir, f'epoch_{epoch}')

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for idx, fn in enumerate(fn_list):
        filename, ext = os.path.splitext(fn)
        if ext == '.mat':
            if tensor_img[idx].device != 'cpu':
                query_data = tensor_img[idx].detach().cpu().numpy()
            else:
                query_data = tensor_img[idx].detach().numpy()

            savemat(os.path.join(save_dir, fn), {f'{filename}': query_data})
            save_img(tensor_img[idx], os.path.join(save_dir, f'{filename}.mat'), gray=False)
        else:
            save_img(tensor_img[idx], os.path.join(save_dir, fn), gray=True)


def save_result_image_loss(save_dir, epoch, fn_list, tensor_img, loss):
    # if epoch is not None:
    if epoch %10==0:
        save_dir = os.path.join(save_dir, f'epoch_{epoch}')

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for idx, fn in enumerate(fn_list):
            filename, ext = os.path.splitext(fn)
            if ext == '.mat':
                if tensor_img[idx].device != 'cpu':
                    query_data = tensor_img[idx].detach().cpu().numpy()
                else:
                    query_data = tensor_img[idx].detach().numpy()

                savemat(os.path.join(save_dir, f'{loss:.6f}_{fn}'), {f'{filename}': query_data})
                save_img(tensor_img[idx], os.path.join(save_dir, f'{loss:.6f}_{filename}.mat'), gray=False)
            else:
                save_img(tensor_img[idx], os.path.join(save_dir, f'{loss:.6f}_{fn}.png'), gray=True)


# 改写：torchvision.utils.save_image,使得可以保存8位灰度图
def save_img(img, img_path, gray=False):

    grid = vutils.make_grid(img, nrow=8, padding=2, pad_value=0,
                            normalize=False, range=None, scale_each=False)
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)

    if gray:
        im.convert('L').save(img_path)  # Gray = 0.29900 * R + 0.58700 * G + 0.11400 * B
    else:
        im.save(img_path)


def save_results_in_file(save_dir, save_filename, fn_list, loss_list):
    save_strings = ''
    for fn, loss in zip(fn_list, loss_list):
        save_strings += f'{fn[0]} {loss}\n'

    with open(os.path.join(save_dir, save_filename), 'w') as f:
        f.write(save_strings)


def create_eval_dir(save_dir):
    if not os.path.exists(save_dir):
        save_dir = os.path.join(save_dir, '0')
        os.makedirs(save_dir)
    else:
        fn_list = list(map(int, os.listdir(save_dir)))
        if len(fn_list) == 0:
            save_dir = os.path.join(save_dir, '0')
        else:
            save_dir = os.path.join(save_dir, str(max(fn_list) + 1))
        os.makedirs(save_dir)
    return save_dir
