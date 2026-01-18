#把 MOT17 的 GT 数据转换成 KalmanNet 的训练格式

# prepare_mot_data.py
import os
import numpy as np
import torch

def load_mot_gt(data_root):
    """
    读取 MOT17 的 gt.txt 文件，转换成训练序列
    """
    # 假设你的 MOT17 数据集在这个路径
    # 如果不是，请修改为你自己的路径 H:\Code\Byte\ByteTrack-main\datasets\mot\train
    train_dir = os.path.join(data_root, 'train') 
    
    all_tracks_obs = [] # 观测 (x, y, a, h)
    all_tracks_gt = []  # 真值 (x, y, a, h)

    seqs = os.listdir(train_dir)
    print(f"Found sequences: {seqs}")

    for seq in seqs:
        if 'FRCNN' not in seq: continue # 只用 FRCNN 的序列
        
        gt_path = os.path.join(train_dir, seq, 'gt', 'gt.txt')
        if not os.path.exists(gt_path): continue
        
        print(f"Processing {seq}...")
        raw_data = np.loadtxt(gt_path, delimiter=',')
        # 格式: frame, id, left, top, width, height, ...
        
        # 按 ID 分组
        ids = np.unique(raw_data[:, 1])
        for target_id in ids:
            track_data = raw_data[raw_data[:, 1] == target_id]
            # 按帧号排序
            track_data = track_data[np.argsort(track_data[:, 0])]
            
            # 提取 x, y, a, h
            # x = left + width/2
            # y = top + height/2
            # a = width / height
            # h = height
            
            x = track_data[:, 2] + track_data[:, 4] / 2
            y = track_data[:, 3] + track_data[:, 5] / 2
            w = track_data[:, 4]
            h = track_data[:, 5]
            
            # 避免除以0
            h[h==0] = 1e-5
            a = w / h
            
            # 组合成 (Seq_Len, 4)
            traj = np.stack((x, y, a, h), axis=1)
            
            # 我们需要把它切分成很多小段，喂给 RNN
            # 比如每段长度 20 帧
            seq_len = 20
            if len(traj) < seq_len: continue
            
            for i in range(0, len(traj) - seq_len, 10): # 步长10，重叠采样
                segment = traj[i : i+seq_len]
                
                # 制造一点噪声作为 Observation (模拟检测误差)
                noise = np.random.randn(*segment.shape) * np.array([5, 5, 0.05, 5]) 
                obs = segment + noise
                
                all_tracks_gt.append(segment)
                all_tracks_obs.append(obs)

    # 转换为 Tensor
    X = torch.tensor(np.array(all_tracks_obs), dtype=torch.float32)
    Y = torch.tensor(np.array(all_tracks_gt), dtype=torch.float32)
    
    print(f"Total samples generated: {len(X)}")
    torch.save((X, Y), 'mot_train_data.pt')
    print("Data saved to mot_train_data.pt")

if __name__ == '__main__':
    # 请修改这里的路径指向你的 datasets/mot
    load_mot_gt('datasets/mot')