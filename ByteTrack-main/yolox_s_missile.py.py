import os
from yolox.exp import Exp as MyExp

class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        # 1. 模型配置 (YOLOX-S)
        self.depth = 0.33
        self.width = 0.50
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # 2. 关键修改：类别数 (导弹只有 1 类)
        self.num_classes = 1

        # 3. 训练参数
        self.max_epoch = 100       # 训练 100 轮
        self.print_interval = 10
        self.eval_interval = 5
        self.save_history_ckpt = False
        self.input_size = (640, 640) # 输入图片大小
        self.test_size = (640, 640)

        # 4. 数据集路径 (对应你 datasets/missile 下的结构)
        self.data_dir = "datasets/missile"
        self.train_ann = "instances_train2017.json"
        self.val_ann = "instances_val2017.json"
        self.train_image_folder = "train2017"
        self.val_image_folder = "val2017"

    def get_dataset(self, cache: bool, cache_type: str = "ram"):
        from yolox.data import COCODataset
        return COCODataset(
            data_dir=self.data_dir,
            json_file=self.train_ann,
            name=self.train_image_folder,
            img_size=self.input_size,
            preproc=self.preproc,
            cache=cache,
            cache_type=cache_type,
        )

    def get_eval_dataset(self, **kwargs):
        from yolox.data import COCODataset
        return COCODataset(
            data_dir=self.data_dir,
            json_file=self.val_ann,
            name=self.val_image_folder,
            img_size=self.test_size,
            preproc=self.preproc,
        )