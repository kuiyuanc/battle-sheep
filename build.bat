@REM Put this file in root directory of this repo
@REM This file wraps your source code and run it

@REM Usage: ./build.bat <game-setting=1> <executable-name-1=Model_1> <agent-to-replace-1=1> <source-code-name-1=Sample>
@REM To pass latter parameters, the former parameters should all be passed because they are positional arguments

@REM Explanation:
@REM 1. Wrap your source code 'Sample.py' and store the executable 'Model_1.exe' into bin
@REM 2. Fix content of 'input.txt' & 'STcpClient.py' for running 'AI_game.exe' ( Replace sample/Sample_1.exe in game play )
@REM 3. Run 'AI_game.exe' with assigned game setting ( default: 1 )

@cls
@echo off

@REM Check if there is an assignment of student ID or not
if "%1"=="" (
    set game_setting=1
) else (
    set game_setting=%1
)

@REM Check if there is any assignment of wrapped executable name
if "%2"=="" (
    set executable1=Model_1
) else (
    set executable1=%2
)

@REM Check if there is any assignment of agent to replace
if "%3"=="" (
    set agent1=1
) else (
    set agent1=%3
)

@REM Check if there is any assignment of name of source code to wrap
if "%4"=="" (
    set source1=Sample
) else (
    set source1=%4
)

cd game_%game_setting%

echo Pre-processing...
py ../fix_input.py %agent1% %executable1%

echo Wrapping source code...
@REM Use the next line (stay unchanged) if you don't install pyinstaller in pipenv
pyinstaller --onefile --dist bin --name %executable1%.exe %source1%.py
@REM Use the next line (comment the previous line & uncomment the next line) if you install pyinstaller in pipenv
@REM pipenv run pyinstaller --onefile --dist bin --name %executable1%.exe %source1%.py

rm *.spec

echo Running game...
start AI_game.exe
