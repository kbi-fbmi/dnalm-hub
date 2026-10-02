# genomic-mcp-client

`GenericMcpClient`: a stdlib-only (no requests/httpx) Python MCP client base class,
factored out of `ntv3_mcp_client.Ntv3McpClient`. Session management (lazy-create,
reuse, transparent re-init on expiry), SSE response parsing, and every tool-wrapper
method (`embed_sequence`, `score_snp`, `compare_sequences`, `get_model_info`,
`get_embedding_layers`, `list_available_checkpoints`, `predict_masked_positions`)
live here once; each service's own `<family>_mcp_client` package is a thin,
zero-logic subclass, since every service in this family exposes the same tool
names and argument shapes (see each server's README for which tools it omits).
