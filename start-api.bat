@echo off
echo Starting ASL Detection API server...
cd /d %~dp0
.\\.venv\Scripts\python.exe backend\api.py --model_path models\asl_model.h5
