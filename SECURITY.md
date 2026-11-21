# Security Policy

## Reporting a Vulnerability

If you discover a security issue (such as unsafe expression evaluation in `anchordoc/adjudicate/calc.py` or credential handling in `anchordoc/runtime/llm.py`), please open a private security advisory on GitHub rather than a public issue.

## Repository Safety Guardrails

- `tools/scan_sensitive.py` scans worktree, staged files, and commit history for private IPs, API keys, and tokens.
- Arithmetic evaluation uses a strict Python AST allow-list (`+`, `-`, `*`, `/`, unary `+/-`, numeric constants) and never calls `eval()` or `exec()`.
