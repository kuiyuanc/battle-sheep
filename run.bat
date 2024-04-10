@REM Put this file in root directory of this repo
@REM This file is for fast running "after" running build.bat

@REM Usage: ./run.bat <game-setting=1> <executable-name-1=Model_1> <agent-to-replace-1=1>
@REM To pass latter parameters, the former parameters should all be passed because they are positional arguments

@REM Explanation:
@REM 1. Adjust 'input.txt' according to assigned agent
@REM 2. Run 'AI_game.exe' with assigned game setting ( default: 1 )

@cls
@echo off

@REM Check if there is an assignment of game setting or not
if "%1"=="" (
    set game_setting=1
) else (
    set game_setting=%1
)

@REM Check if there is an assignment of executable or not
if "%2"=="" (
    set executable1=Model_1
) else (
    set executable1=%2
)

@REM Check if there is an assignment of agnet to be replaced with or not
if "%3"=="" (
    set agent1=1
) else (
    set agent1=%3
)

cd game_%game_setting%

echo Pre-processing...
py ../fix_input.py %agent1% %executable1%

echo Running game...
start AI_game.exe
