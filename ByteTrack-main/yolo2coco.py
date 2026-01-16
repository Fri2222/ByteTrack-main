import os
import json
import shutil
import cv2
from tqdm import tqdm

# ================= 配置区域 =================
# 1. 你现在的 YOLOv8 数据集路径 (请核对你的实际路径)
YOLO_ROOT = r"H:\Code\Byte\ByteTrack-main\datasets\missle\armament.v7i.yolov8"

# 2. 目标 COCO 数据集路径 (脚本会自动创建这些文件夹)
COCO_ROOT = r"H:\Code\Byte\ByteTrack-main\datasets\missile"

# 3. 你的类别名称 (必须与 data.yaml 里的一致，顺序不能错)
# 如果 data.yaml 里只有一个类，通常是 'missile'
CLASSES = ["missile"]


# ===========================================

def convert_yolo_to_coco(subset, src_dir, dst_img_dir, dst_json_path):
    """
    subset: 'train' 或 'valid'
    src_dir: YOLO数据的子目录，例如 YOLO_ROOT/train
    dst_img_dir: 目标图片目录，例如 COCO_ROOT/train2017
    dst_json_path: 目标JSON路径
    """

    # 确保目标目录存在
    os.makedirs(dst_img_dir, exist_ok=True)
    os.makedirs(os.path.dirname(dst_json_path), exist_ok=True)

    images = []
    annotations = []
    categories = [{"id": i + 1, "name": name} for i, name in enumerate(CLASSES)]

    # YOLO图片和标签路径
    img_dir_src = os.path.join(src_dir, "images")
    label_dir_src = os.path.join(src_dir, "labels")

    img_files = [f for f in os.listdir(img_dir_src) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    ann_id = 1

    print(f"正在处理 {subset} 集，共 {len(img_files)} 张图片...")

    for img_id, filename in enumerate(tqdm(img_files)):
        # 1. 复制图片
        src_img_path = os.path.join(img_dir_src, filename)
        dst_img_path = os.path.join(dst_img_dir, filename)
        shutil.copy2(src_img_path, dst_img_path)

        # 2. 读取图片宽高
        img = cv2.imread(src_img_path)
        height, width = img.shape[:2]

        # 添加图片信息
        images.append({
            "id": img_id + 1,
            "file_name": filename,
            "height": height,
            "width": width
        })

        # 3. 读取对应的 txt 标签
        label_filename = os.path.splitext(filename)[0] + ".txt"
        label_path = os.path.join(label_dir_src, label_filename)

        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                lines = f.readlines()

            for line in lines:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue  # 跳过空行或坏数据

                cls_id = int(parts[0])
                val_data = list(map(float, parts[1:]))

                # === 修改开始：自动兼容检测和分割格式 ===
                if len(val_data) == 4:
                    # 情况A: 标准检测格式 (x, y, w, h)
                    cx, cy, w, h = val_data
                else:
                    # 情况B: 分割/多边形格式 (x1, y1, x2, y2, ...)
                    # 我们需要算出它的外接矩形
                    xs = val_data[0::2]  # 取偶数位作为x
                    ys = val_data[1::2]  # 取奇数位作为y
                    min_x, max_x = min(xs), max(xs)
                    min_y, max_y = min(ys), max(ys)

                    w = max_x - min_x
                    h = max_y - min_y
                    cx = min_x + w / 2
                    cy = min_y + h / 2
                # === 修改结束 ===

                # 转换坐标: center -> top-left (绝对坐标)
                abs_w = w * width
                abs_h = h * height
                abs_x = (cx * width) - (abs_w / 2)
                abs_y = (cy * height) - (abs_h / 2)

                annotations.append({
                    "id": ann_id,
                    "image_id": img_id + 1,
                    "category_id": cls_id + 1,
                    "bbox": [abs_x, abs_y, abs_w, abs_h],
                    "area": abs_w * abs_h,
                    "iscrowd": 0
                })
                ann_id += 1

    # 保存 JSON
    coco_format = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }

    with open(dst_json_path, "w") as f:
        json.dump(coco_format, f)

    print(f"完成！已生成: {dst_json_path}")


if __name__ == "__main__":
    # 转换训练集
    convert_yolo_to_coco(
        "train",
        os.path.join(YOLO_ROOT, "train"),  # YOLO train 文件夹
        os.path.join(COCO_ROOT, "train2017"),  # 目标 train2017 文件夹
        os.path.join(COCO_ROOT, "annotations", "instances_train2017.json")
    )

    # 转换验证集
    convert_yolo_to_coco(
        "valid",
        os.path.join(YOLO_ROOT, "valid"),  # YOLO valid 文件夹
        os.path.join(COCO_ROOT, "val2017"),  # 目标 val2017 文件夹
        os.path.join(COCO_ROOT, "annotations", "instances_val2017.json")
    )