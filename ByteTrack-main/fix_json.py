import json
import os

# 定义你的标注文件路径
base_path = r"datasets\mot\annotations"
files_to_fix = ["val_half.json", "train_half.json", "test.json", "train.json"]


def fix_coco_json(file_name):
    file_path = os.path.join(base_path, file_name)

    if not os.path.exists(file_path):
        print(f"跳过: {file_name} (文件不存在)")
        return

    print(f"正在修复: {file_name} ...")

    try:
        with open(file_path, 'r') as f:
            data = json.load(f)

        # 核心修复逻辑：如果没有 'info' 字段，就加一个空的
        if 'info' not in data:
            data['info'] = {"description": "Fixed by user", "version": "1.0", "year": 2025}

            with open(file_path, 'w') as f:
                json.dump(data, f)
            print(f"✅ 修复成功: {file_name}")
        else:
            print(f"无需修复: {file_name} (已有 info 字段)")

    except Exception as e:
        print(f"❌ 修复失败 {file_name}: {e}")


# 执行修复
for f in files_to_fix:
    fix_coco_json(f)