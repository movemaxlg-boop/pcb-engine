@echo off
echo ============================================================
echo PCB ENGINE - AI SETUP
echo ============================================================
echo.
echo This will help you set up a FREE AI provider for PCB Engine.
echo.
echo Choose your AI provider:
echo   1. Groq (Recommended - Fast, Free)
echo   2. Google Gemini (Free tier)
echo   3. Ollama (Local, completely free)
echo   4. Skip (use rule-based system)
echo.
set /p choice="Enter choice (1-4): "

if "%choice%"=="1" goto groq
if "%choice%"=="2" goto gemini
if "%choice%"=="3" goto ollama
if "%choice%"=="4" goto skip
goto invalid

:groq
echo.
echo === GROQ SETUP ===
echo.
echo 1. Go to: https://console.groq.com/keys
echo 2. Sign up (free, no credit card)
echo 3. Create an API key
echo.
set /p key="Paste your Groq API key (gsk_...): "
if "%key%"=="" goto invalid_key
setx GROQ_API_KEY "%key%"
echo.
echo [SUCCESS] Groq API key saved!
echo Restart your terminal for changes to take effect.
goto test

:gemini
echo.
echo === GOOGLE GEMINI SETUP ===
echo.
echo 1. Go to: https://makersuite.google.com/app/apikey
echo 2. Sign in with Google
echo 3. Create an API key
echo.
set /p key="Paste your Gemini API key: "
if "%key%"=="" goto invalid_key
setx GEMINI_API_KEY "%key%"
echo.
echo [SUCCESS] Gemini API key saved!
echo Restart your terminal for changes to take effect.
goto test

:ollama
echo.
echo === OLLAMA SETUP ===
echo.
echo 1. Download from: https://ollama.ai
echo 2. Install and run
echo 3. Open a new terminal and run: ollama run llama3.1
echo.
echo PCB Engine will auto-detect Ollama when running.
pause
goto end

:skip
echo.
echo Using rule-based system (no AI).
echo You can set up AI later by running this script again.
goto end

:invalid
echo Invalid choice. Please run again.
goto end

:invalid_key
echo No key entered. Please run again.
goto end

:test
echo.
echo Testing connection...
python -c "from ai_connector import create_ai_connector; ai = create_ai_connector('groq' if '%choice%'=='1' else 'gemini'); r = ai.chat('Say OK'); print('SUCCESS!' if r.success else f'Failed: {r.error}')"
echo.
pause

:end
echo.
echo To start PCB Engine Web:
echo   cd web
echo   python server.py
echo.
echo Then open: http://localhost:8080
echo.
pause
