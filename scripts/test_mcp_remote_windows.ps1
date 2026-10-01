param(
    [string]$ServerUrl = "http://kbi-cs2.fbmi.cvut.cz:8000",
    [string]$McpPath = "/mcp",
    [string]$Checkpoint = "100m-pre",
    [string]$Sequence = "ACGTACGTACGT",
    [string]$ProtocolVersion = "2025-06-18"
)

$ErrorActionPreference = "Stop"

function Write-Step {
    param([string]$Text)
    Write-Host "[$((Get-Variable step -Scope Script -ErrorAction SilentlyContinue).Value)/4] $Text"
}

$script:step = 1

$authToken = $env:MCP_AUTH_TOKEN
$headers = @{}
if (-not [string]::IsNullOrWhiteSpace($authToken)) {
    $headers["Authorization"] = "Bearer $authToken"
}

Write-Host "[1/4] Health check: $ServerUrl/health"
$health = Invoke-WebRequest -Uri "$ServerUrl/health" -Method Get -UseBasicParsing
Write-Host "Health status: $($health.StatusCode)"
Write-Host "Health body: $($health.Content)"

Write-Host "[2/4] MCP initialize"
$initPayload = @{
    jsonrpc = "2.0"
    id = "init-1"
    method = "initialize"
    params = @{
        protocolVersion = $ProtocolVersion
        capabilities = @{}
        clientInfo = @{
            name = "ntv3-mcp-win-smoke"
            version = "1.0"
        }
    }
} | ConvertTo-Json -Depth 10 -Compress

$initResp = Invoke-WebRequest -Uri "$ServerUrl$McpPath" -Method Post -Headers $headers -ContentType "application/json" -Body $initPayload -UseBasicParsing
$sessionId = $initResp.Headers["mcp-session-id"]
if ([string]::IsNullOrWhiteSpace($sessionId)) {
    throw "Initialize response did not include mcp-session-id."
}
Write-Host "Session: $sessionId"

$headersWithSession = @{}
foreach ($k in $headers.Keys) { $headersWithSession[$k] = $headers[$k] }
$headersWithSession["Mcp-Session-Id"] = $sessionId

$initializedPayload = '{"jsonrpc":"2.0","method":"notifications/initialized","params":{}}'
Invoke-WebRequest -Uri "$ServerUrl$McpPath" -Method Post -Headers $headersWithSession -ContentType "application/json" -Body $initializedPayload -UseBasicParsing | Out-Null

function Parse-SseJson {
    param([string]$SseBody)
    $line = ($SseBody -split "`n" | Where-Object { $_ -like "data:*" } | Select-Object -First 1)
    if ([string]::IsNullOrWhiteSpace($line)) {
        throw "No 'data:' line found in SSE response."
    }
    $jsonText = $line.Substring(5).Trim()
    return $jsonText | ConvertFrom-Json -Depth 20
}

Write-Host "[3/4] list_available_checkpoints"
$listPayload = '{"jsonrpc":"2.0","id":"call-list-1","method":"tools/call","params":{"name":"list_available_checkpoints","arguments":{}}}'
$listResp = Invoke-WebRequest -Uri "$ServerUrl$McpPath" -Method Post -Headers $headersWithSession -ContentType "application/json" -Body $listPayload -UseBasicParsing
$listObj = Parse-SseJson -SseBody $listResp.Content

if ($listObj.result.isError -eq $true) {
    Write-Host "List checkpoints returned error:" -ForegroundColor Red
    $listObj.result.content | ForEach-Object { Write-Host $_.text }
    exit 2
}

$modelNames = @()
if ($listObj.result.structuredContent -and $listObj.result.structuredContent.result) {
    $modelNames = $listObj.result.structuredContent.result | ForEach-Object { $_.name }
} elseif ($listObj.result.content) {
    foreach ($item in $listObj.result.content) {
        try {
            $obj = $item.text | ConvertFrom-Json -Depth 10
            if ($obj.name) { $modelNames += $obj.name }
        } catch {
            # Ignore parse misses in fallback mode.
        }
    }
}

if ($modelNames.Count -gt 0) {
    Write-Host "Available models:"
    $modelNames | ForEach-Object { Write-Host " - $_" }
} else {
    Write-Host "Could not parse model names; raw tools response follows:"
    Write-Host $listResp.Content
}

Write-Host "[4/4] embed_sequence checkpoint=$Checkpoint sequence=$Sequence"
$embedPayload = @{
    jsonrpc = "2.0"
    id = "call-embed-1"
    method = "tools/call"
    params = @{
        name = "embed_sequence"
        arguments = @{
            sequence = $Sequence
            checkpoint = $Checkpoint
            pooling = "mean"
        }
    }
} | ConvertTo-Json -Depth 15 -Compress

$embedResp = Invoke-WebRequest -Uri "$ServerUrl$McpPath" -Method Post -Headers $headersWithSession -ContentType "application/json" -Body $embedPayload -UseBasicParsing
$embedObj = Parse-SseJson -SseBody $embedResp.Content

if ($embedObj.result.isError -eq $true) {
    Write-Host "Embedding call returned error:" -ForegroundColor Yellow
    $embedObj.result.content | ForEach-Object { Write-Host $_.text }
    Write-Host "Tip: nastav platný HF_TOKEN na serveru a zkontroluj přístup ke gated NTv3 repu."
    exit 3
}

Write-Host "Embedding call succeeded."
if ($embedObj.result.structuredContent) {
    $checkpointUsed = $embedObj.result.structuredContent.checkpoint
    $poolingUsed = $embedObj.result.structuredContent.pooling
    Write-Host " checkpoint: $checkpointUsed"
    Write-Host " pooling: $poolingUsed"
}

Write-Host "Done."
