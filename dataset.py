# Author: Lishuo Pan 2020/4/18

import os.path
import matplotlib.patches as patches
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import torch
import torchvision.transforms.functional
import torch.nn.functional as F
from torch.utils.data import DataLoader
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')               # No display


class BuildDataset(torch.utils.data.Dataset):
    def __init__(self, path, augmentation=True):
        """

        :param path: the path to the dataset root: /workspace/data or XXX/data/SOLO
        :param argumentation: if True, perform horizontal flipping argumentation
        """
        self.augmentation = augmentation
        imgs_path, masks_path, labels_path, bboxes_path = path

        self.images_h5 = h5py.File(imgs_path, 'r')
        self.masks_h5 = h5py.File(masks_path, 'r')
        self.labels_all = np.load(labels_path, allow_pickle=True)
        self.bboxes_all = np.load(bboxes_path, allow_pickle=True)

        n_objects_img = [len(self.labels_all[i]) for i in range(
            len(self.labels_all))]
        self.mask_offset = np.cumsum(n_objects_img)
        self.mask_offset = np.concatenate([np.array([0]), self.mask_offset])

    # output:
    # transed_img
    # label
    # transed_mask
    # transed_bbox
    # Note: For data augmentation, number of items is 2 * N_images
    def __getitem__(self, index):
        # images
        img_np = self.images_h5['data'][index] / 255.0  # (3, 300, 400)
        img = torch.tensor(img_np, dtype=torch.float)

        label = torch.tensor(self.labels_all[index], dtype=torch.long)
        mask_offset_s = self.mask_offset[index]
        mask_list = []
        for i in range(len(label)):
            mask_np = self.masks_h5['data'][mask_offset_s + i] * 1.0
            mask_tmp = torch.tensor(mask_np, dtype=torch.float)
            mask_list.append(mask_tmp)
        mask = torch.stack(mask_list)

        bbox_np = self.bboxes_all[index]
        bbox = torch.tensor(bbox_np, dtype=torch.float)
        transed_img, transed_mask, transed_bbox = self.pre_process_batch(
            img, mask, bbox)
        if self.augmentation and (np.random.rand(1).item() > 0.5):
            assert transed_img.ndim == 3
            assert transed_mask.ndim == 3
            transed_img = torch.flip(transed_img, dims=[2])
            transed_mask = torch.flip(transed_mask, dims=[2])
            transed_bbox_new = transed_bbox.clone()
            transed_bbox_new[:, 0] = 1 - transed_bbox[:, 2]
            transed_bbox_new[:, 2] = 1 - transed_bbox[:, 0]
            transed_bbox = transed_bbox_new

            assert torch.all(transed_bbox[:, 0] < transed_bbox[:, 2])

        # check flag
        assert transed_img.shape == (3, 800, 1088)
        assert transed_bbox.shape[0] == transed_mask.shape[0]
        return transed_img, label, transed_mask, transed_bbox

    def __len__(self):
        return self.images_h5['data'].shape[0]

    # This function take care of the pre-process of img,mask,bbox
    # in the input mini-batch
    # input:
        # img: 3*300*400
        # mask: 3*300*400
        # bbox: n_box*4
    def pre_process_batch(self, img, mask, bbox):
        assert isinstance(img, torch.Tensor)
        assert isinstance(mask, torch.Tensor)
        assert isinstance(bbox, torch.Tensor)
        img = self.interp2d(img, 800, 1066)
        img = torchvision.transforms.functional.normalize(img, mean=[0.485, 0.456, 0.406],
                                                          std=[0.229, 0.224, 0.225])
        img = F.pad(img, [11, 11])

        mask = self.interp2d(mask, 800, 1066)
        mask = F.pad(mask, [11, 11])

        bbox_normed = torch.zeros_like(bbox)
        for i in range(bbox.shape[0]):
            bbox_normed[i, 0] = bbox[i, 0] / 400.0
            bbox_normed[i, 1] = bbox[i, 1] / 300.0
            bbox_normed[i, 2] = bbox[i, 2] / 400.0
            bbox_normed[i, 3] = bbox[i, 3] / 300.0
        assert torch.max(bbox_normed) <= 1.0
        assert torch.min(bbox_normed) >= 0.0
        bbox = bbox_normed

        # check flag
        assert img.shape == (3, 800, 1088)
        assert bbox.shape[0] == mask.shape[0]
        return img, mask, bbox

    @staticmethod
    def interp2d(x, H, W):
        assert isinstance(x, torch.Tensor)
        channels = x.shape[0]
        x_expanded = torch.unsqueeze(x, 0)
        x_expanded = torch.unsqueeze(x_expanded, 0)

        result = F.interpolate(x_expanded, (channels, H, W))
        result = result.squeeze(0)
        result = result.squeeze(0)
        return result


class BuildDataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size, shuffle, num_workers):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers

    # output:

        # img: (bz, 3, 800, 1088)
        # label_list: list, len:bz, each (n_obj,)
        # transed_mask_list: list, len:bz, each (n_obj, 800,1088)
        # transed_bbox_list: list, len:bz, each (n_obj, 4)
    def collect_fn(self, batch):
        transed_img_list = []
        label_list = []
        transed_mask_list = []
        transed_bbox_list = []
        for transed_img, label, transed_mask, transed_bbox in batch:
            transed_img_list.append(transed_img)
            label_list.append(label)
            transed_mask_list.append(transed_mask)
            transed_bbox_list.append(transed_bbox)
        return torch.stack(transed_img_list, dim=0), label_list, transed_mask_list, transed_bbox_list

    def loader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers,
                          collate_fn=self.collect_fn)


# Visualize debugging
if __name__ == '__main__':

    imgs_path = 'hw3_mycocodata_img_comp_zlib.h5'
    masks_path = 'hw3_mycocodata_mask_comp_zlib.h5'
    labels_path = 'hw3_mycocodata_labels_comp_zlib.npy'
    bboxes_path = 'hw3_mycocodata_bboxes_comp_zlib.npy'

    os.makedirs("test_figure", exist_ok=True)

    paths = [imgs_path, masks_path, labels_path, bboxes_path]
    # load the data into data.Dataset
    dataset = BuildDataset(paths)

    # Visualize debugging
    # --------------------------------------------
    # build the dataloader
    # set 20% of the dataset as the training data
    full_size = len(dataset)
    train_size = int(full_size * 0.8)
    test_size = full_size - train_size
    # random split the dataset into training and testset
    # set seed
    torch.random.manual_seed(1)
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size])
    # push the randomized training data into the dataloader

    batch_size = 2
    train_build_loader = BuildDataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    train_loader = train_build_loader.loader()
    test_build_loader = BuildDataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = test_build_loader.loader()

    mask_color_list = ["jet", "ocean", "Spectral", "spring", "cool"]
    # convert to rgb color list
    rgb_color_list = []
    for color_str in mask_color_list:
        color_map = cm.ScalarMappable(cmap=color_str)
        rgb_value = np.array(color_map.to_rgba(0))[:3]
        rgb_color_list.append(rgb_value)

    # loop the image
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for iter, data in enumerate(train_loader, 0):

        img, label, mask, bbox = [data[i] for i in range(len(data))]
        # check flag
        assert img.shape == (batch_size, 3, 800, 1088)
        assert len(mask) == batch_size

        label = [label_img.to(device) for label_img in label]
        mask = [mask_img.to(device) for mask_img in mask]
        bbox = [bbox_img.to(device) for bbox_img in bbox]

        # plot the origin img
        for i in range(batch_size):
            # TODO: plot images with annotations
            fig, ax = plt.subplots(1)
            alpha = 0.15
            img_vis = img[i]
            img_vis = img_vis.permute((1, 2, 0)).cpu().numpy()

            for obj_i, obj_mask in enumerate(mask[i], 0):
                obj_label = label[i][obj_i]

                rgb_color = rgb_color_list[obj_label - 1]
                obj_mask_np = np.stack(
                    [obj_mask.cpu().numpy(), obj_mask.cpu().numpy(), obj_mask.cpu().numpy()], axis=2)
                img_vis[obj_mask_np != 0] = (
                    (1-alpha) * rgb_color + alpha * img_vis)[obj_mask_np != 0]

            img_vis = np.clip(img_vis, 0, 1)
            ax.imshow(img_vis)

            for obj_i, obj_bbox_normed in enumerate(bbox[i], 0):
                bbox_res = torch.tensor(obj_bbox_normed, dtype=torch.float)
                bbox_res[0] = obj_bbox_normed[0] * 1066 + 11
                bbox_res[1] = obj_bbox_normed[1] * 800
                bbox_res[2] = obj_bbox_normed[2] * 1066 + 11
                bbox_res[3] = obj_bbox_normed[3] * 800
                obj_bbox = bbox_res
                obj_w = obj_bbox[2] - obj_bbox[0]
                obj_h = obj_bbox[3] - obj_bbox[1]
                rect = patches.Rectangle(
                    (obj_bbox[0], obj_bbox[1]), obj_w, obj_h, linewidth=1, edgecolor='r', facecolor='none')
                ax.add_patch(rect)

            plt.savefig(
                "./test_figure/visualtrainset_{}_{}_.png".format(iter, i))
            plt.show()

        if iter == 40:
            break
