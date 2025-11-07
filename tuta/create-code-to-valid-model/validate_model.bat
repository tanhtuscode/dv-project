@echo off
echo ============================================
echo SkateboardML Model Validation
echo Created by: Tran Anh Tu
echo ============================================
echo.

cd /d "%~dp0"

echo Checking Python environment...
python --version

echo.
echo Running model validation...
echo.

python run_validation.py

echo.
echo Validation completed!
echo Check the validation_results folder for detailed reports.
echo.
pause