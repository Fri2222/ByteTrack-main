####################################################################################################################

项目名称:bytetrack
环境名称:bytetrack


####################################################################################################################


B. 输入特征 (Input Features)传统 MB KF 及其变体利用底层统计特性的知识来计算 KG 。为了让神经网络学会这种计算，必须向其提供能够捕获评估 KG 所需信息的输入特征 。$\mathcal{K}_{t}$ 对观测值和状态过程统计特性的依赖关系表明，为了跟踪它，在每个时间步 $t\in\mathcal{T}$，应该向 RNN 提供包含观测值 $y_{t}$ 和状态估计信息的输入 。因此，我们将以下四个量作为 RNN 的输入特征：F1 观测差值 (Observation difference)： $\Delta\tilde{y}_{t}=y_{t}-y_{t-1}$ F2 新息差值 (Innovation difference)： $\Delta y_{t}=y_{t}-\hat{y}_{t|t-1}$ F3 前向演化差值 (Forward evolution difference)： $\Delta\tilde{x}_{t}=\hat{x}_{t|t}-\hat{x}_{t-1|t-1}$ 。该量表示两个连续的后验状态估计之间的差值。对于时间步 $t$，可用的特征是 $\Delta\tilde{x}_{t-1}$ 。F4 前向更新差值 (Forward update difference)： $\Delta\hat{x}_{t}=\hat{x}_{t|t}-\hat{x}_{t|t-1}$ 。即后验状态估计与先验状态估计之间的差值。同样，对于时间步 $t$，我们使用的是 $\Delta\hat{x}_{t-1}$ 。特征 F1 和 F3 封装了有关状态演化过程的信息，而特征 F2 和 F4 封装了状态估计中的不确定性 。这种差分操作去除了序列中可预测的成分，使得差分时间序列主要受到我们真正希望学习的“噪声统计特性”的影响 。广泛的经验评估表明，特征的特定组合选择取决于具体的任务问题 。我们发现较好的特征组合通常是 {F1, F2, F4} 以及 {F1, F3, F4} 
可能有用的改进信息




1.训练yolox_s权重文件
python tools/train.py -f exps/example/mot/yolox_s_mot17_half.py -d 1 -b 4 --fp16 -o -c pretrained/yolox_s.pth

2.用训练完成的权重文件识别视频demo
python tools/demo_track.py video -f exps/example/mot/yolox_s_mot17_half.py -c pretrained/bytetrack_s_mot17.pth.tar --path "视频地址" --fp16 --fuse --save_result

3.使用已训练的yolox_S权重文件，并且使用原始KF测试参数
python tools/track.py -f exps/example/mot/yolox_s_mot17_half.py -c pretrained/yolox_s.pth -b 1 -d 1 --fp16 --fuse

4.使用已训练的yolox_S权重文件，并且使用改进KF测试参数
python tools/track.py -f exps/example/mot/yolox_s_mot17_half.py -c pretrained/yolox_s.pth -b 1 -d 1 --fp16 --fuse --eval




####################################################################################################################
anaconda命令

1.conda info --envs

2.conda activate bytetrack

3.cd H:\Code\Byte\ByteTrack-main\ByteTrack-main

####################################################################################################################
git命令

1. 添加所有文件到暂存区
	git add .

2.提交文件到本地仓库
	git commit -m "这里写备注"
	
3.将本地代码推送到 GitHub
	git push -u origin <分支名>
	
4.查看当前分支
	git branch
	
5.创建分支
	git branch <分支名>
	
6.切换分支
	git checkout <分支名>