import csv
import os
import numpy as np
import scipy.ndimage
import scipy.special
import skimage
import skimage.filters
import skvideo.io
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Resize, Compose, ToTensor, Normalize


def get_mgrid(sidelen, dim=2):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.'''
    if isinstance(sidelen, int):
        sidelen = dim * (sidelen,)

    if dim == 2:
        pixel_coords = np.stack(np.mgrid[:sidelen[0], :sidelen[1]], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[0, :, :, 0] = pixel_coords[0, :, :, 0] / (sidelen[0] - 1)
        pixel_coords[0, :, :, 1] = pixel_coords[0, :, :, 1] / (sidelen[1] - 1)
    elif dim == 3:
        pixel_coords = np.stack(np.mgrid[:sidelen[0], :sidelen[1], :sidelen[2]], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[..., 0] = pixel_coords[..., 0] / max(sidelen[0] - 1, 1)
        pixel_coords[..., 1] = pixel_coords[..., 1] / (sidelen[1] - 1)
        pixel_coords[..., 2] = pixel_coords[..., 2] / (sidelen[2] - 1)
    else:
        raise NotImplementedError('Not implemented for dim=%d' % dim)

    pixel_coords -= 0.5
    pixel_coords *= 2.
    pixel_coords = torch.Tensor(pixel_coords).view(-1, dim)
    return pixel_coords


def lin2img(tensor, image_resolution=None):
    batch_size, num_samples, channels = tensor.shape
    if image_resolution is None:
        width = np.sqrt(num_samples).astype(int)
        height = width
    else:
        height = image_resolution[0]
        width = image_resolution[1]

    return tensor.permute(0, 2, 1).view(batch_size, channels, height, width)

class Camera(Dataset):
    def __init__(self, downsample_factor=1):
        super().__init__()
        self.downsample_factor = downsample_factor
        self.img = Image.fromarray(skimage.data.camera())
        self.img_channels = 1

        if downsample_factor > 1:
            size = (int(512 / downsample_factor),) * 2
            self.img_downsampled = self.img.resize(size, Image.ANTIALIAS)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        if self.downsample_factor > 1:
            return self.img_downsampled
        else:
            return self.img
        
class ImageFile(Dataset):
    def __init__(self, filename):
        super().__init__()
        self.img = Image.open(filename)
        self.img_channels = len(self.img.mode)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.img


class CelebA(Dataset):
    def __init__(self, split, downsampled=False):
        # SIZE (178 x 218)
        super().__init__()
        assert split in ['train', 'test', 'val'], "Unknown split"

        self.root = '/media/data3/awb/CelebA/kaggle/img_align_celeba/img_align_celeba'
        self.img_channels = 3
        self.fnames = []

        with open('/media/data3/awb/CelebA/kaggle/list_eval_partition.csv', newline='') as csvfile:
            rowreader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for row in rowreader:
                if split == 'train' and row[1] == '0':
                    self.fnames.append(row[0])
                elif split == 'val' and row[1] == '1':
                    self.fnames.append(row[0])
                elif split == 'test' and row[1] == '2':
                    self.fnames.append(row[0])

        self.downsampled = downsampled

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        path = os.path.join(self.root, self.fnames[idx])
        img = Image.open(path)
        if self.downsampled:
            width, height = img.size  # Get dimensions

            s = min(width, height)
            left = (width - s) / 2
            top = (height - s) / 2
            right = (width + s) / 2
            bottom = (height + s) / 2
            img = img.crop((left, top, right, bottom))
            img = img.resize((32, 32))

        return img


class Implicit2DWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset, sidelength=None, compute_diff=None):

        if isinstance(sidelength, int):
            sidelength = (sidelength, sidelength)
        self.sidelength = sidelength

        self.transform = Compose([
            Resize(sidelength),
            ToTensor(),
            Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))
        ])

        self.compute_diff = compute_diff
        self.dataset = dataset
        self.mgrid = get_mgrid(sidelength)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img = self.transform(self.dataset[idx])

        if self.compute_diff == 'gradients':
            img *= 1e1
            gradx = scipy.ndimage.sobel(img.numpy(), axis=1).squeeze(0)[..., None]
            grady = scipy.ndimage.sobel(img.numpy(), axis=2).squeeze(0)[..., None]
        elif self.compute_diff == 'laplacian':
            img *= 1e4
            laplace = scipy.ndimage.laplace(img.numpy()).squeeze(0)[..., None]
        elif self.compute_diff == 'all':
            gradx = scipy.ndimage.sobel(img.numpy(), axis=1).squeeze(0)[..., None]
            grady = scipy.ndimage.sobel(img.numpy(), axis=2).squeeze(0)[..., None]
            laplace = scipy.ndimage.laplace(img.numpy()).squeeze(0)[..., None]

        img = img.permute(1, 2, 0).view(-1, self.dataset.img_channels)

        in_dict = {'idx': idx, 'coords': self.mgrid}
        gt_dict = {'img': img}

        if self.compute_diff == 'gradients':
            gradients = torch.cat((torch.from_numpy(gradx).reshape(-1, 1),
                                   torch.from_numpy(grady).reshape(-1, 1)),
                                  dim=-1)
            gt_dict.update({'gradients': gradients})

        elif self.compute_diff == 'laplacian':
            gt_dict.update({'laplace': torch.from_numpy(laplace).view(-1, 1)})

        elif self.compute_diff == 'all':
            gradients = torch.cat((torch.from_numpy(gradx).reshape(-1, 1),
                                   torch.from_numpy(grady).reshape(-1, 1)),
                                  dim=-1)
            gt_dict.update({'gradients': gradients})
            gt_dict.update({'laplace': torch.from_numpy(laplace).view(-1, 1)})

        return in_dict, gt_dict

    def get_item_small(self, idx):
        img = self.transform(self.dataset[idx])
        spatial_img = img.clone()
        img = img.permute(1, 2, 0).view(-1, self.dataset.img_channels)

        gt_dict = {'img': img}

        return spatial_img, img, gt_dict


class ImageGeneralizationWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset, test_sparsity=None, train_sparsity_range=(10, 200), generalization_mode=None):
        self.dataset = dataset
        self.sidelength = dataset.sidelength
        self.mgrid = dataset.mgrid
        self.test_sparsity = test_sparsity
        self.train_sparsity_range = train_sparsity_range
        self.generalization_mode = generalization_mode

    def __len__(self):
        return len(self.dataset)

    # update the sparsity of the images used in testing
    def update_test_sparsity(self, test_sparsity):
        self.test_sparsity = test_sparsity

    # generate the input dictionary based on the type of model used for generalization
    def get_generalization_in_dict(self, spatial_img, img, idx):
        # case where we use the convolutional encoder for generalization, either testing or training
        if self.generalization_mode == 'conv_cnp' or self.generalization_mode == 'conv_cnp_test':
            if self.test_sparsity == 'full':
                img_sparse = spatial_img
            elif self.test_sparsity == 'half':
                img_sparse = spatial_img
                img_sparse[:, 16:, :] = 0.
            else:
                if self.generalization_mode == 'conv_cnp_test':
                    num_context = int(self.test_sparsity)
                else:
                    num_context = int(
                        torch.empty(1).uniform_(self.train_sparsity_range[0], self.train_sparsity_range[1]).item())
                mask = spatial_img.new_empty(
                    1, spatial_img.size(1), spatial_img.size(2)).bernoulli_(p=num_context / np.prod(self.sidelength))
                img_sparse = mask * spatial_img
            in_dict = {'idx': idx, 'coords': self.mgrid, 'img_sparse': img_sparse}
        # case where we use the set encoder for generalization, either testing or training
        elif self.generalization_mode == 'cnp' or self.generalization_mode == 'cnp_test':
            if self.test_sparsity == 'full':
                in_dict = {'idx': idx, 'coords': self.mgrid, 'img_sub': img, 'coords_sub': self.mgrid}
            elif self.test_sparsity == 'half':
                in_dict = {'idx': idx, 'coords': self.mgrid, 'img_sub': img[:512, :], 'coords_sub': self.mgrid[:512, :]}
            else:
                if self.generalization_mode == 'cnp_test':
                    subsamples = int(self.test_sparsity)
                    rand_idcs = np.random.choice(img.shape[0], size=subsamples, replace=False)
                    img_sparse = img[rand_idcs, :]
                    coords_sub = self.mgrid[rand_idcs, :]
                    in_dict = {'idx': idx, 'coords': self.mgrid, 'img_sub': img_sparse, 'coords_sub': coords_sub}
                else:
                    subsamples = np.random.randint(self.train_sparsity_range[0], self.train_sparsity_range[1])
                    rand_idcs = np.random.choice(img.shape[0], size=self.train_sparsity_range[1], replace=False)
                    img_sparse = img[rand_idcs, :]
                    coords_sub = self.mgrid[rand_idcs, :]

                    rand_idcs_2 = np.random.choice(img_sparse.shape[0], size=subsamples, replace=False)
                    ctxt_mask = torch.zeros(img_sparse.shape[0], 1)
                    ctxt_mask[rand_idcs_2, 0] = 1.

                    in_dict = {'idx': idx, 'coords': self.mgrid, 'img_sub': img_sparse, 'coords_sub': coords_sub,
                               'ctxt_mask': ctxt_mask}
        else:
            in_dict = {'idx': idx, 'coords': self.mgrid}

        return in_dict

    def __getitem__(self, idx):
        spatial_img, img, gt_dict = self.dataset.get_item_small(idx)
        in_dict = self.get_generalization_in_dict(spatial_img, img, idx)
        return in_dict, gt_dict

