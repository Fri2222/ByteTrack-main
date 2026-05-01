@echo off
chcp 65001 >nul

echo ==================================================
echo 🚀 开始全自动执行 KalmanNet 序列长度消融实验
echo ==================================================

:: 使用绝对路径强制激活基础环境，然后再切换到 bytetrack
call F:\Software\Anaconda\Scripts\activate.bat F:\Software\Anaconda
call conda activate bytetrack

:: 检查是否激活成功
echo 当前 Python 环境：
where python

FOR /L %%L IN (15, 5, 65) DO (
    echo.
    echo ^>^>^> 正在启动实验: Seq-Len = %%L ...
    
    python train_kalmannet.py --seq-len %%L --seq-step 5 --epochs 60
    
    if errorlevel 1 (
        echo.
        echo ❌ 训练出错！在截断长度 %%L 处异常退出，自动化脚本已中止。
        goto :error
    )
    
    :: 👇 [核心新增]：训练成功后，立刻在 exp0031 文件夹中将生成的权重重命名
    if exist "H:\Code\Byte\ByteTrack-main\ByteTrack-main\pretrained\exp0031\kalmannet_best.pth" (
        ren "H:\Code\Byte\ByteTrack-main\ByteTrack-main\pretrained\exp0031\kalmannet_best.pth" "kalmannet_best_len%%L_step5.pth"
        echo 📦 本轮权重已安全封存为: kalmannet_best_len%%L_step5.pth
    ) else (
        echo ⚠️ 警告：未找到生成的权重文件！
    )
)

echo.
echo ==================================================
echo 🎉 所有消融实验执行完毕！权重已全部保存在 exp0031 文件夹中。
echo ==================================================
pause
exit /b

:error
pause
exit /b