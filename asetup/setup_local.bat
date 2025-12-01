@echo off
REM Quick setup for local 3090 machine
REM Updates existing nd_unc environment

echo Setting up nd_unc environment for local testing...

call conda activate nd_unc
if %errorlevel% neq 0 (
    echo ERROR: nd_unc environment not found!
    echo Please create it first or activate manually
    pause
    exit 1
)

echo Installing/updating packages...
pip install -r requirements.txt

echo.
echo Running test...
cd ..
python asetup/test_setup.py

echo.
echo Setup complete!
pause
