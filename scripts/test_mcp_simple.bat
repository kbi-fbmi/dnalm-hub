@echo off
setlocal

set "SERVER_URL=%~1"
if "%SERVER_URL%"=="" set "SERVER_URL=http://kbi-cs2.fbmi.cvut.cz:8000"
set "TOKEN=local-test-token"
set "HEADERS_FILE=%TEMP%\mcp_init_headers.txt"
set "EMBED_FILE=%TEMP%\mcp_embed_response.sse"
set "SID_FILE=%TEMP%\mcp_session_id.txt"

echo [1/3] initialize session
curl -sS -D "%HEADERS_FILE%" -o nul -X POST "%SERVER_URL%/mcp" -H "Authorization: Bearer %TOKEN%" -H "Content-Type: application/json" -H "Accept: application/json, text/event-stream" --data "{\"jsonrpc\":\"2.0\",\"id\":\"init-1\",\"method\":\"initialize\",\"params\":{\"protocolVersion\":\"2025-06-18\",\"capabilities\":{},\"clientInfo\":{\"name\":\"ntv3-mcp-simple-bat\",\"version\":\"1.0\"}}}"

powershell -NoProfile -Command "$h=Get-Content '%HEADERS_FILE%'; $sid=($h | Where-Object { $_ -match '^mcp-session-id:' } | ForEach-Object { $_.Split(':',2)[1].Trim() }); Set-Content -NoNewline '%SID_FILE%' $sid"
set /p SESSION_ID=<"%SID_FILE%"

if "%SESSION_ID%"=="" (
  echo Failed to get mcp-session-id.
  type "%HEADERS_FILE%"
  exit /b 1
)

echo Session: %SESSION_ID%
echo [2/3] initialized notification
curl -sS -o nul -X POST "%SERVER_URL%/mcp" -H "Authorization: Bearer %TOKEN%" -H "Mcp-Session-Id: %SESSION_ID%" -H "Content-Type: application/json" -H "Accept: application/json, text/event-stream" --data "{\"jsonrpc\":\"2.0\",\"method\":\"notifications/initialized\",\"params\":{}}"

echo [3/3] simple embed_sequence query
curl -sS -o "%EMBED_FILE%" -X POST "%SERVER_URL%/mcp" -H "Authorization: Bearer %TOKEN%" -H "Mcp-Session-Id: %SESSION_ID%" -H "Content-Type: application/json" -H "Accept: application/json, text/event-stream" --data "{\"jsonrpc\":\"2.0\",\"id\":\"embed-1\",\"method\":\"tools/call\",\"params\":{\"name\":\"embed_sequence\",\"arguments\":{\"sequence\":\"ACGTACGTACGT\",\"checkpoint\":\"100m-pre\",\"pooling\":\"mean\"}}}"

echo ---- response ----
type "%EMBED_FILE%"
echo.
echo Done.
