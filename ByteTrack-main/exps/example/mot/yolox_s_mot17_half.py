import os
import torch
import torch.distributed as dist
from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        # ================== 1. 核心修复：添加数据路径 ==================
        self.data_dir = "./datasets"

        # ================== 2. 模型参数 (YOLOX-S) ==================
        self.depth = 0.33
        self.width = 0.50
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # ================== 3. 数据与类别 ==================
        # self.num_classes = 80  <-- 之前的修改，现在要删掉或注释
        self.num_classes = 1  # <-- 改回 1，匹配 MOT17 专用权重
        self.input_size = (800, 1440)
        self.test_size = (800, 1440)
        self.train_ann = "train_half.json"
        self.val_ann = "val_half.json"
        self.seed = None

        # ================== 4. 训练超参数 ==================
        self.max_epoch = 80
        self.print_interval = 20
        self.eval_interval = 5
        self.test_conf = 0.01
        self.nmsthre = 0.7
        self.no_aug_epochs = 10
        self.basic_lr_per_img = 0.001 / 64.0
        self.warmup_epochs = 1

        # ================== 5. 数据增强参数 (防止后续报错) ==================
        self.degrees = 10.0
        self.translate = 0.1
        self.scale = (0.1, 2)
        self.shear = 2.0
        self.perspective = 0.0
        self.enable_mixup = True

    # ---------------- 训练数据加载器 ---------------- #
    def get_data_loader(self, batch_size, is_distributed, no_aug=False, cache_img=False):
        from yolox.data import (
            MOTDataset,
            TrainTransform,
            YoloBatchSampler,
            DataLoader,
            InfiniteSampler,
            MosaicDetection,
        )

        dataset = MOTDataset(
            data_dir=os.path.join(self.data_dir, "mot"),
            json_file=self.train_ann,
            name='train',
            img_size=self.input_size,
            preproc=TrainTransform(
                rgb_means=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_labels=500,
            ),
        )

        dataset = MosaicDetection(
            dataset,
            mosaic=not no_aug,
            img_size=self.input_size,
            preproc=TrainTransform(
                rgb_means=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_labels=1000,
            ),
            degrees=self.degrees,
            translate=self.translate,
            scale=self.scale,
            shear=self.shear,
            perspective=self.perspective,
            enable_mixup=self.enable_mixup,
        )

        self.dataset = dataset

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()

        sampler = InfiniteSampler(
            len(self.dataset), seed=self.seed if self.seed else 0
        )

        batch_sampler = YoloBatchSampler(
            sampler=sampler,
            batch_size=batch_size,
            drop_last=False,
            input_dimension=self.input_size,
            mosaic=not no_aug,
        )

        dataloader = DataLoader(
            self.dataset,
            batch_sampler=batch_sampler,
            num_workers=4,
            pin_memory=True,
        )

        return dataloader

    # ---------------- 验证数据加载器 ---------------- #
    def get_eval_loader(self, batch_size, is_distributed, testdev=False, legacy=False):
        from yolox.data import MOTDataset, ValTransform

        valdataset = MOTDataset(
            data_dir=os.path.join(self.data_dir, "mot"),
            json_file=self.val_ann,
            img_size=self.test_size,
            name='train',
            preproc=ValTransform(
                rgb_means=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        )

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()
            sampler = torch.utils.data.distributed.DistributedSampler(
                valdataset, shuffle=False
            )
        else:
            sampler = torch.utils.data.SequentialSampler(valdataset)

        dataloader = torch.utils.data.DataLoader(
            valdataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=4,
            pin_memory=True,
            shuffle=False
        )

        return dataloader