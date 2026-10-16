@echo off
rem Stop a running local-upscale job (variants or finish) and free the GPU.
rem
rem Normally you just close the job's window ("local-upscale ... - close this window to stop it").
rem Use this when there is no window to close, or from a script. It stops only jobs started from
rem THIS install, closes their windows, and stops local-image's server too if the job was a
rem variants run (which would have stopped it when done). local-image's own stop.cmd stops that
rem server alone. Exit status: 0 stopped or nothing running, 1 some processes survived.
setlocal
if not exist "%~dp0.venv\Scripts\python.exe" (
  echo No .venv here: run install.ps1 first.
  exit /b 1
)
"%~dp0.venv\Scripts\python.exe" "%~dp0scripts\upscale.py" stop
