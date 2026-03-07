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