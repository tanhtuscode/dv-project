@echo off
echo Starting SkateboardML Web Application...
echo.
echo Make sure you have activated your Python environment and installed requirements:
echo   pip install -r config\requirements_web.txt
echo.
echo The web application will be available at: http://localhost:5000
echo.
cd app
python web_app.py
pause