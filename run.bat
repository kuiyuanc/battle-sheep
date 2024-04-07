@REM Put this file in root directory of this repo
@REM This file is for fast running "after" running build.bat
@REM Usage: ./run.bat or ./run.bat

@cls
@echo off

@REM Check if there is an assignment of game setting or not
if "%1"=="" (
    set game_setting=1
) else (
    set game_setting=%1
)

cd game_%game_setting%

echo Running game...
start AI_game.exe
