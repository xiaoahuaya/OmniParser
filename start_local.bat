@echo off
echo ========================================
echo  OmniParser 本地控制模式启动脚本
echo ========================================
echo.

REM 激活 conda 环境
call conda activate OmniParser

echo [1/2] 启动 OmniParser Server...
start "OmniParser Server" cmd /k "cd /d F:\ideacode\OmniParser\omnitool\omniparserserver && python -m omniparserserver --som_model_path ../../weights/icon_detect/model.pt --caption_model_name florence2 --caption_model_path ../../weights/icon_caption_florence --device cuda"

echo 等待 OmniParser Server 启动...
timeout /t 10 /nobreak

echo [2/2] 启动 Gradio UI (本地模式)...
start "Gradio UI" cmd /k "cd /d F:\ideacode\OmniParser\omnitool\gradio && python app.py --omniparser_server_url localhost:8000 --local"

echo.
echo ========================================
echo  服务启动完成！
echo  - OmniParser Server: http://localhost:8000
echo  - Gradio UI: http://localhost:7888
echo ========================================
echo.
pause
