from PIL import Image
import sys
import numpy as np
import random
import os
from collections import defaultdict
from .vid import VIDDataset, make_vid_transforms
# from mega_core.config import cfg
import datasets.transforms as T
import torch
from .torchvision_datasets import CocoDetection

class VIDCOCODataset(CocoDetection):

    def __init__(self, cfg, img_folder, ann_file, transforms,is_train=True, return_masks=False, cache_mode=False, local_rank=0, local_size=1):
        super().__init__(img_folder, ann_file,
                                            cache_mode=cache_mode, local_rank=local_rank, local_size=local_size)
        self.cfg = cfg
        self.max_offset = 12
        self.min_offset = -12
        self.ref_num_local = 2

        self.test_with_one_img = False
        self.test_ref_nums = 2
        self.test_max_offset = 12
        self.test_min_offset = -12
        self.is_train = is_train
        self.transforms = transforms
        
        self.ids = self.ids[::cfg.DATASET.frame_extraction]

        # self.frame_seg_len = 
        videos_imgs = defaultdict(list)
        for item in self.coco.imgs.items():
            img_dict = item[-1]
            video_name = os.path.dirname(img_dict["file_name"])
            videos_imgs[video_name].append(img_dict)
        
        self.frame_seg_len = list(self.ids) # 指示了当前idx视频帧对应的最后一帧的id
        for i, idx in enumerate(self.ids):
            # import pdb;pdb.set_trace()
            img_dict = self.coco.imgs[idx]
            video_name = os.path.dirname(img_dict["file_name"])
            self.frame_seg_len[i] = videos_imgs[video_name][0]["id"] + len(videos_imgs[video_name])


        if cfg is not None:

            self.test_with_one_img = cfg.TEST.test_with_one_img
            self.test_ref_nums = cfg.TEST.test_ref_nums
            self.test_max_offset = cfg.TEST.test_max_offset
            self.test_min_offset = cfg.TEST.test_min_offset


        if cfg is not None:
            self.max_offset = cfg.DATASET.max_offset
            self.min_offset = cfg.DATASET.min_offset
            self.ref_num_local = cfg.DATASET.ref_num_local


    def __getitem__(self, idx):
        if self.is_train:
            return self._get_train(idx)
        else:
            return self._get_test(idx)
    
    def _pre_target(self,anno,img):

        height, width = img["height"],img["width"]
        boxes = [x["bbox"] for x in anno]
        labels = [x["category_id"] for x in anno]
        areas = [x["area"] for x in anno]
        
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 3] = boxes[:, 3] + boxes[:, 1]
        boxes[:, 2] = boxes[:, 2] + boxes[:, 0]
        # # boxes[:, 2:] += boxes[:, :2]  # ??? bug
        # boxes[:, 0::2].clamp_(min=0, max=width)
        # boxes[:, 1::2].clamp_(min=0, max=height)
        # keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        # boxes = boxes[keep]

        # labels = torch.tensor(labels[keep], dtype=torch.int64)
        # areas = torch.tensor(areas[keep], dtype=torch.float32)
        size = torch.tensor([height, width])  #  rechecked, is h w
        
        # target = BoxList(anno["boxes"].reshape(-1, 4), (width, height), mode="xyxy")
        # target.add_field("labels", anno["labels"])
        target = {"boxes": boxes, "orig_size": size, "size": size, "labels": labels, "areas": areas,
                  "image_id": torch.tensor(img["id"])}  # todo image_id
        
        return target
    
    def _get_train(self, idx):

        coco = self.coco
        img_id = self.ids[idx]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = self._pre_target(coco.loadAnns(ann_ids),coco.imgs[img_id])

        path = coco.imgs[img_id]['file_name']

        img = self.get_image(path)
        # if self.transforms is not None:
        #     img, target = self.transforms(img, target)

        # if a video dataset
        img_refs_l = []
        target_refs_l = []
        # img_refs_m = []
        # img_refs_g = []
        # if hasattr(self, "pattern"):
        offsets = np.random.choice(self.max_offset - self.min_offset + 1,
                                    self.ref_num_local, replace=False) + self.min_offset
        for i in range(len(offsets)):
            ref_id = min(max(img_id + offsets[i], 0), self.frame_seg_len[idx] - 1)
            ref_filename = coco.imgs[ref_id]['file_name']
            img_ref = self.get_image(ref_filename)
            img_refs_l.append(img_ref)
            target_ref = self._pre_target(coco.loadAnns(coco.getAnnIds(imgIds=ref_id)),coco.imgs[ref_id])
            target_refs_l.append(target_ref)
            assert (os.path.dirname(ref_filename) == os.path.dirname(path)), "find other video frame,please check vidcoco.py"

        # else:
        #     for i in range(self.ref_num_local):
        #         img_refs_l.append(img.copy())
        #         target_refs_l.append(target.copy())

        p_dict = None
        #
        if self.transforms is not None:
            img, target = self.transforms(img, target, p_dict)
            for i in range(len(img_refs_l)):
                img_refs_l[i], _ = self.transforms(img_refs_l[i], None, p_dict)
        images = {}
        images["cur"] = img  # to make a list
        images["ref_l"] = img_refs_l

        return images, target

    def _get_test(self, idx):
        coco = self.coco
        img_id = self.ids[idx]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = self._pre_target(coco.loadAnns(ann_ids),coco.imgs[img_id])

        path = coco.imgs[img_id]['file_name']

        img = self.get_image(path)

        # give the current frame a category. 0 for start, 1 for normal
        frame_id = int(os.path.splitext(os.path.basename(path))[0].split("_")[-1])
        frame_category = 0
        if frame_id != 0:
            frame_category = 1

        if self.test_with_one_img:
            img_refs_l = []
            # reading other images of the queue (not necessary to be the last one, but last one here)
            ref_id = min(self.frame_seg_len[idx] - 1, img_id +self.max_offset)
            ref_filename = coco.imgs[ref_id]['file_name']
            img_ref = self.get_image(ref_filename)
            img_refs_l.append(img_ref)
            assert (os.path.dirname(ref_filename) == os.path.dirname(path)), "find other video frame,please check vidcoco.py"
        else:
            img_refs_l = self.get_ref_imgs(img_id)

        img_refs_g = []

        # target = target.clip_to_image(remove_empty=True)

        if self.transforms is not None:
            img, target = self.transforms(img, target)
            for i in range(len(img_refs_l)):
                img_refs_l[i], _ = self.transforms(img_refs_l[i], None)
            for i in range(len(img_refs_g)):
                img_refs_g[i], _ = self.transforms(img_refs_g[i], None)

        images = {}
        images["cur"] = img
        images["ref_l"] = img_refs_l

        return images, target

    def get_ref_imgs(self, idx):
        img_id = self.ids[idx]
        filename = self.coco.imgs[img_id]['file_name']
        # img = self.get_image(filename)

        # give the current frame a category. 0 for start, 1 for normal
        frame_id = int(os.path.splitext(os.path.basename(filename))[0].split("_")[-1])
        ref_id_list = []
        ref_start_id = idx + self.test_min_offset
        ref_end_id = idx + self.test_max_offset

        interval = (ref_end_id - ref_start_id) // (self.test_ref_nums - 1)

        for i in range(ref_start_id, ref_end_id + 1, interval):
            # print(i)
            ref_id_list.append(min(img_id + max(0, i), self.frame_seg_len[idx] - 1))

        # for i in range(ref_start_id, ref_end_id + 1):
        #     ref_id_list.append(min(max(0, i), self.frame_seg_len[idx] - 1))

        img_refs_l = []

        for ref_id in ref_id_list:
            # print(ref_id)
            ref_filename = self.coco.imgs[ref_id]['file_name']
            img_ref = self.get_image(ref_filename)
            img_refs_l.append(img_ref)
            assert (os.path.dirname(ref_filename) == os.path.dirname(filename)), "find other video frame,please check vidcoco.py"
        return img_refs_l



def build_vitcoco_transforms(is_train):
    # todo fixme add data augmantation
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    # if is_train:
    #     return T.Compose([
    #         T.RandomHorizontalFlip(),
    #         T.RandomSelect(
    #             T.RandomResize(scales, max_size=1333, id=0),
    #             T.Compose([
    #                 T.RandomResize([400, 500, 600], id=1),
    #                 # T.RandomSizeCrop(384, 600),  # todo if cropping is neccessary? 'cause it may lead to no bbox in the image. In current version, cur and ref images are cropped using different parameters.
    #                 T.RandomResize(scales, max_size=1333, id=2),
    #             ])
    #         ),
    #         normalize,
    #     ])

    # if image_set == 'val':
    return T.Compose([
        T.RandomResize([800], max_size=1333),
        normalize,
    ])
    # return transform



def build_vidcoco(image_set, cfg,  transforms=build_vitcoco_transforms(True)):

    is_train = (image_set == 'train')
    if is_train:
        dataset = VIDCOCODataset(
        cfg = cfg, 
        img_folder = cfg.DATASET.train_img_folder, 
        ann_file = cfg.DATASET.train_ann_file, 
        transforms = transforms,
        is_train = is_train
        )
    else:
        dataset = VIDCOCODataset(
        cfg = cfg, 
        img_folder = cfg.DATASET.val_img_folder, 
        ann_file = cfg.DATASET.val_ann_file, 
        transforms = transforms,
        is_train = is_train
        )

    return dataset

def build_detmulti(image_set, cfg):

    is_train = (image_set == 'train')
    assert is_train is True  # no validation dataset
    dataset = VIDCOCODataset(
    image_set = "DET_train_30classes",
    img_dir = "/dataset/public/ilsvrc2015/Data/DET",
    anno_path = "/dataset/public/ilsvrc2015/Annotations/DET",
    img_index = "/data1/wanghan20/Prj/VODETR/datasets/split_file/DET_train_30classes.txt",
    transforms=build_vitcoco_transforms(True),
    )
    return dataset