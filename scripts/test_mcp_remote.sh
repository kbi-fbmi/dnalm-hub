#!/usr/bin/env bash
set -euo pipefail

SERVER_URL="${1:-http://kbi-cs2.fbmi.cvut.cz:8000}"
MCP_PATH="${MCP_PATH:-/mcp}"
CHECKPOINT="${CHECKPOINT:-100m-pre}"
SEQUENCE="${SEQUENCE:-ACGTACGTACGT}"
PROTO_VERSION="${PROTO_VERSION:-2025-06-18}"
AUTH_TOKEN="${MCP_AUTH_TOKEN:-}"

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

AUTH_HEADERS=()
if [[ -n "$AUTH_TOKEN" ]]; then
  AUTH_HEADERS=(-H "Authorization: Bearer ${AUTH_TOKEN}")
fi

post_mcp() {
  local payload="$1"
  local body_out="$2"
  local header_out="$3"
  shift 3
  curl -sS -D "$header_out" -o "$body_out" \
    -X POST "${SERVER_URL}${MCP_PATH}" \
    -H "Content-Type: application/json" \
    -H "Accept: application/json, text/event-stream" \
    "${AUTH_HEADERS[@]}" \
    "$@" \
    --data "$payload"
}

extract_data_json() {
  # MCP streamable-HTTP replies as SSE lines: "data: {json}".
  sed -n 's/^data: //p' "$1" | head -n 1
}

echo "[1/4] Health check: ${SERVER_URL}/health"
if ! HEALTH_BODY="$(curl -fsS "${SERVER_URL}/health")"; then
  echo "Health check failed."
  exit 1
fi
echo "Health OK: ${HEALTH_BODY}"

echo "[2/4] MCP initialize"
INIT_PAYLOAD='{"jsonrpc":"2.0","id":"init-1","method":"initialize","params":{"protocolVersion":"'"${PROTO_VERSION}"'","capabilities":{},"clientInfo":{"name":"ntv3-mcp-curl-smoke","version":"1.0"}}}'
post_mcp "$INIT_PAYLOAD" "$TMP_DIR/init.body" "$TMP_DIR/init.headers"

SESSION_ID="$(awk -F': ' 'tolower($1)=="mcp-session-id" {gsub("\r", "", $2); print $2}' "$TMP_DIR/init.headers")"
if [[ -z "$SESSION_ID" ]]; then
  echo "Initialize did not return mcp-session-id. Headers:"
  cat "$TMP_DIR/init.headers"
  echo "Body:"
  cat "$TMP_DIR/init.body"
  exit 1
fi
echo "Session: ${SESSION_ID}"

post_mcp \
  '{"jsonrpc":"2.0","method":"notifications/initialized","params":{}}' \
  "$TMP_DIR/initialized.body" \
  "$TMP_DIR/initialized.headers" \
  -H "Mcp-Session-Id: ${SESSION_ID}" >/dev/null

echo "[3/4] list_available_checkpoints"
LIST_PAYLOAD='{"jsonrpc":"2.0","id":"call-list-1","method":"tools/call","params":{"name":"list_available_checkpoints","arguments":{}}}'
post_mcp "$LIST_PAYLOAD" "$TMP_DIR/list.body" "$TMP_DIR/list.headers" -H "Mcp-Session-Id: ${SESSION_ID}"
LIST_JSON="$(extract_data_json "$TMP_DIR/list.body")"

if command -v jq >/dev/null 2>&1; then
  echo "$LIST_JSON" | jq -r '.result.content[]?.text' | jq -r '.name' | sed 's/^/- /'
else
  echo "jq not found; raw response:"
  cat "$TMP_DIR/list.body"
fi

echo "[4/4] embed_sequence checkpoint=${CHECKPOINT} sequence=${SEQUENCE}"
EMBED_PAYLOAD='{"jsonrpc":"2.0","id":"call-embed-1","method":"tools/call","params":{"name":"embed_sequence","arguments":{"sequence":"'"${SEQUENCE}"'","checkpoint":"'"${CHECKPOINT}"'","pooling":"mean"}}}'
post_mcp "$EMBED_PAYLOAD" "$TMP_DIR/embed.body" "$TMP_DIR/embed.headers" -H "Mcp-Session-Id: ${SESSION_ID}"
EMBED_JSON="$(extract_data_json "$TMP_DIR/embed.body")"

if command -v jq >/dev/null 2>&1; then
  IS_ERROR="$(echo "$EMBED_JSON" | jq -r '.result.isError // false')"
  if [[ "$IS_ERROR" == "true" ]]; then
    echo "Embedding call returned an error:"
    echo "$EMBED_JSON" | jq -r '.result.content[]?.text // "(no error text)"'
    exit 2
  fi
  echo "Embedding call succeeded (showing metadata):"
  echo "$EMBED_JSON" | jq '{id, hasResult: (.result != null), contentPreview: (.result.content[0].text // "")[:200]}'
else
  echo "jq not found; raw embedding response:"
  cat "$TMP_DIR/embed.body"
fi

echo "Done."
