@echo off
chcp 65001 >nul

echo ==================================================
echo 🚀 开始全自动评估 KalmanNet 序列长度消融实验权重
echo ==================================================

:: 1. 激活您的 Anaconda 基础环境和 bytetrack 虚拟环境
call F:\Software\Anaconda\Scripts\activate.bat F:\Software\Anaconda
call conda activate bytetrack

:: 2. 创建一个专门存放评估结果的文件夹
if not exist "eval_logs" mkdir "eval_logs"

:: 3. 开始循环测试 (15 到 65，步长 5)
FOR /L %%L IN (15, 5, 65) DO (
    echo.
    echo ==================================================
    echo ^>^>^> [ Seq-Len = %%L ] 正在准备评估...
    echo ==================================================
    
    :: 👇 [修改] 检查 exp0031 文件夹下对应的权重文件是否存在
    if exist "H:\Code\Byte\ByteTrack-main\ByteTrack-main\pretrained\exp0031\kalmannet_best_len%%L_step5.pth" (
        
        :: 👇 [修改] 把 exp0031 里的权重复制为跟踪器默认读取的名字
        copy /Y "H:\Code\Byte\ByteTrack-main\ByteTrack-main\pretrained\exp0031\kalmannet_best_len%%L_step5.pth" "pretrained\kalmannet_best.pth" >nul
        echo ✅ 已成功从 exp0031 加载权重: kalmannet_best_len%%L_step5.pth
        
        :: 运行推理，生成跟踪结果
        echo ⏳ 正在运行测试集推理 track.py ，请稍候...
        python tools/track.py -f exps/example/mot/yolox_s_mot17_half.py -c pretrained/bytetrack_s_mot17.pth.tar -b 1 -d 1 --fp16 --fuse
        
        :: 运行评测脚本，并将结果重定向保存到日志文件中
        echo 📊 正在计算指标 eval_custom.py ...
        python eval_custom.py > eval_logs\eval_len%%L.txt 2>&1
        
        :: 将刚刚保存的日志文件内容直接打印到屏幕上
        echo --------------------------------------------------
        echo 📝 截断长度 %%L 的最终评估结果如下：
        type eval_logs\eval_len%%L.txt
        echo --------------------------------------------------
        
        echo 🎉 长度 %%L 评估完成！结果已永久保存至: eval_logs\eval_len%%L.txt
        
    ) else (
        :: 👇 [修改] 找不到时的报错信息也加上完整路径提示
        echo ⚠️ 找不到权重文件 H:\Code\Byte\ByteTrack-main\ByteTrack-main\pretrained\exp0031\kalmannet_best_len%%L_step5.pth，自动跳过！
    )
)

echo.
echo ==================================================
echo 🏆 所有消融实验权重评估完毕！
echo 您随时可以进入 eval_logs 文件夹查看和复制所有历史记录。
echo ==================================================
pause
exit /b