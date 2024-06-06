# -*- coding: utf-8 -*-
"""
@Time ： 2023/10/18 17:15
@Auth ： Pengcheng Zheng
@File ：three_image_dataset.py
@IDE ：PyCharm
@Motto：ABC(Always Be Coding)
"""
from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.data.data_util import three_paths_from_folder, paired_paths_from_lmdb, paired_paths_from_meta_info_file
from basicsr.data.transforms import augment, three_random_crop
from basicsr.utils import FileClient, bgr2ycbcr, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY

@DATASET_REGISTRY.register()
class ThreeImageDataset(data.Dataset):
    """Three image dataset for joint image deblur and sr .

    Read L_blur, LR, GT image.

    There are three modes:

    1. **lmdb**: Use lmdb files. If opt['io_backend'] == lmdb.
    2. **meta_info_file**: Use meta information file to generate paths. \
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. **folder**: Scan folders to generate paths. The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
        dataroot_gt (str): Data root path for gt.
        dataroot_lq (str): Data root path for lq.
        meta_info_file (str): Path for meta information file.
        io_backend (dict): IO backend type and other kwarg.
        filename_tmpl (str): Template for each filename. Note that the template excludes the file extension.
            Default: '{}'.
        gt_size (int): Cropped patched size for gt patches.
        use_hflip (bool): Use horizontal flips.
        use_rot (bool): Use rotation (use vertical flip and transposing h and w for implementation).
        scale (bool): Scale, which will be added automatically.
        phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(ThreeImageDataset, self).__init__()
        self.opt = opt
        # print("zpc")
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.gt_folder, self.lr_folder, self.lr_blur_folder = opt['dataroot_gt'], opt['dataroot_lr'], opt[
            'dataroot_lr_blur']  # gt->model lr->deblur lr_blur->input
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']
            self.paths = paired_paths_from_lmdb([self.lq_folder, self.gt_folder], ['lq', 'gt'])
        elif 'meta_info_file' in self.opt and self.opt['meta_info_file'] is not None:
            self.paths = paired_paths_from_meta_info_file([self.lq_folder, self.gt_folder], ['lq', 'gt'],
                                                          self.opt['meta_info_file'], self.filename_tmpl)
        else:
            self.paths = three_paths_from_folder([self.lr_folder, self.gt_folder, self.lr_blur_folder],
                                                 ['lr', 'gt', 'lr_blur'], self.filename_tmpl)

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']
        # Load gt lr and lr_blur images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        img_gt = imfrombytes(img_bytes, float32=True)
        lr_path = self.paths[index]['lr_path']
        img_bytes = self.file_client.get(lr_path, 'lr')
        img_lr = imfrombytes(img_bytes, float32=True)
        lr_blur_path = self.paths[index]['lr_blur_path']
        img_bytes = self.file_client.get(lr_blur_path, 'lr_blur')
        img_lr_blur = imfrombytes(img_bytes, float32=True)
        ######################################################
        #         print(img_gt.shape)
        #         import matplotlib.pyplot as plt
        #         plt.imshow(img_gt)
        #         print(img_lr.shape)
        #         plt.imshow(img_lr)
        #         print(img_lr_blur.shape)
        ######################################################
        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # random crop
            img_gt, img_lr, img_lr_blur = three_random_crop(img_gt, img_lr, img_lr_blur, gt_size, scale, gt_path)
            # flip, rotation
            img_gt, img_lr, img_lr_blur = augment([img_gt, img_lr, img_lr_blur], self.opt['use_hflip'],
                                                  self.opt['use_rot'])

        # color space transform
        if 'color' in self.opt and self.opt['color'] == 'y':
            img_gt = bgr2ycbcr(img_gt, y_only=True)[..., None]
            img_lr = bgr2ycbcr(img_lr, y_only=True)[..., None]
            img_lr_blur = bgr2ycbcr(img_lr_blur, y_only=True)[..., None]

        # crop the unmatched GT images during validation or testing, especially for SR benchmark datasets
        # TODO: It is better to update the datasets, rather than force to crop
        if self.opt['phase'] != 'train':
            img_gt = img_gt[0:img_lr.shape[0] * scale, 0:img_lr.shape[1] * scale, :]

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lr, img_lr_blur = img2tensor([img_gt, img_lr, img_lr_blur], bgr2rgb=True, float32=True)

        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lr, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)
            normalize(img_lr_blur, self.mean, self.std, inplace=True)

        return {'lr': img_lr, 'gt': img_gt, 'lr_blur': img_lr_blur, 'lr_path': lr_path, 'gt_path': gt_path,
                'lr_blur_path': lr_blur_path}

    def __len__(self):
        return len(self.paths)
