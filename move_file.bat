@echo off
setlocal

:: Source file path
set "source=D:\Documents\AI\BookSearchArchive\post_documents.txt"

:: Destination directory
set "destination=D:\Documents\AI\GitCompareProject"

:: Move the file, overwriting if it exists
move /Y "%source%" "%destination%"

:: Check if the move was successful
if errorlevel 1 (
    echo Error: Failed to move the file.
) else (
    echo File moved successfully.
)

endlocal
pause