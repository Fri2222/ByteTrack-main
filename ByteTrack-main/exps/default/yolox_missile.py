import os
from yolox.exp import Exp as MyExp
from yolox.data.datasets import COCODataset  # 这里直接导入

class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        # 模型大小配置 (对应你下载的 yolox_s.pth)
        self.depth = 0.33
        self.width = 0.50
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # 🎯 核心设置：你的专属类别数量
        self.num_classes = 1

        # 📂 数据集路径设置
        # 注意：这里假设你已经把 H:\Dataset\Missle\Missile Subset.v3i.coco
        # 复制或软链接到了 ByteTrack-main\datasets 目录下，并改名叫 missile_coco
        self.data_dir = "datasets/missile_coco"

        # 请根据你实际数据集里的 json 文件名修改下面两行
        self.train_ann = "train/instances_train2017"
        self.val_ann = "valid/instances_val2017"

        # 训练参数微调 (可根据你的显卡配置修改)
        self.max_epoch = 100
        self.data_num_workers = 4
        self.eval_interval = 5