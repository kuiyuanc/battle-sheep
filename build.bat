@echo off

set agent_to_replace=1
set game_setting=1
set student_ID=110550035

cd game_%game_setting%

echo Wrapping source code...
pipenv run pyinstaller --onefile --dist . --name %student_ID%.exe Sample.py

echo Pre-processing...
py ../fix_input.py %agent_to_replace% %student_ID%

echo Running game...
start AI_game.exe
