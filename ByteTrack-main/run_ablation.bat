@echo off
chcp 65001 >nul
:: 设置控制台为 UTF-8 编码，防止中文乱码

echo ==================================================
echo 🚀 开始全自动执行 KalmanNet 序列长度消融实验
echo ==================================================

:: 1. 自动激活您的 Anaconda 基础环境和 bytetrack 虚拟环境
:: (请根据您自己电脑上 Anaconda 的实际安装路径修改下面这一行，通常在 C盘 或 F盘)
:: call F:\Software\Anaconda\Scripts\activate.bat F:\Software\Anaconda
call conda activate bytetrack

:: 2. 开始循环 (从 15 开始，每次加 5，直到 65)
FOR /L %%L IN (15, 5, 65) DO (
    echo.
    echo ^>^>^> 正在启动实验: Seq-Len = %%L ...
    
    python train_kalmannet.py --seq-len %%L --seq-step 5 --epochs 60
    
    :: 错误拦截
    if errorlevel 1 (
        echo.
        echo ❌ 训练出错！在截断长度 %%L 处异常退出，自动化脚本已中止。
        goto :error
    )
)

echo.
echo ==================================================
echo 🎉 所有消融实验执行完毕！
echo ==================================================
pause
exit /b

:error
pause
exit /b