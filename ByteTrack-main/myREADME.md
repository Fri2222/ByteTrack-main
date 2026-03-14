####################################################################################################################

项目名称:bytetrack
环境名称:bytetrack


####################################################################################################################

1.训练yolox_s权重文件
python tools/train.py -f exps/example/mot/yolox_s_mot17_half.py -d 1 -b 4 --fp16 -o -c pretrained/yolox_s.pth

2.用训练完成的权重文件识别视频demo
python tools/demo_track.py video -f exps/example/mot/yolox_s_mot17_half.py -c YOLOX_outputs/yolox_s_mot17_half/best_ckpt.pth.tar --path test.mp4 --fp16 --fuse --save_result

3.使用已训练的yolox_S权重文件，并且使用原始KF测试参数
python tools/track.py -f exps/example/mot/yolox_s_mot17_half.py -c pretrained/yolox_s.pth -b 1 -d 1 --fp16 --fuse

4.使用已训练的yolox_S权重文件，并且使用改进KF测试参数
python tools/track.py -f exps/example/mot/yolox_s_mot17_half.py -c pretrained/yolox_s.pth -b 1 -d 1 --fp16 --fuse --eval

5.使用Bytetrack的MOT官方权重(不使用3.4.)
python tools/track.py -f exps/example/mot/yolox_s_mot17_half.py -c pretrained/bytetrack_s_mot17.pth.tar -b 1 -d 1 --fp16 --fuse


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
	