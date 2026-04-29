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
    echo ^>^>^> [ Seq-Len = %%L ] 正在准备评估...
    
    :: 检查对应的权重文件是否存在
    if exist "pretrained\kalmannet_best_len%%L_step5.pth" (
        
        :: 【移花接木】：把当前循环的权重复制为跟踪器默认读取的名字
        copy /Y "pretrained\kalmannet_best_len%%L_step5.pth" "pretrained\kalmannet_best.pth" >nul
        echo     ✅ 已加载权重: kalmannet_best_len%%L_step5.pth
        
        :: 【第一步】：运行推理，生成跟踪结果 (去掉了括号防报错)
        echo     ⏳ 正在运行测试集推理 track.py ，请稍候...
        python tools/track.py -f exps/example/mot/yolox_s_mot17_half.py -c pretrained/bytetrack_s_mot17.pth.tar -b 1 -d 1 --fp16 --fuse
        
        :: 【第二步】：运行评测脚本，并将结果重定向保存到日志文件中 (去掉了括号防报错)
        echo     📊 正在计算指标并保存日志 eval_custom.py ...
        python eval_custom.py > eval_logs\eval_len%%L.txt 2>&1
        
        echo     🎉 长度 %%L 评估完成！结果已保存至: eval_logs\eval_len%%L.txt
        
    ) else (
        echo     ⚠️ 找不到权重文件 pretrained\kalmannet_best_len%%L_step5.pth，自动跳过！
    )
)

echo.
echo ==================================================
echo 🏆 所有消融实验权重评估完毕！
echo 请打开 eval_logs 文件夹查看每一个长度的具体指标。
echo ==================================================
pause
exit /b