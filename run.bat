@cls
@echo off

@REM Check if the first argument (%1) is not empty
if "%1"=="" (
    set game_setting=1
) else (
    set game_setting=%1
)

cd game_%game_setting%

echo Running game...
start AI_game.exe
