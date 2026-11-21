#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HOOK="$ROOT/.git/hooks/pre-commit"
cat > "$HOOK" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
python3 tools/scan_sensitive.py --staged
EOF
chmod +x "$HOOK"
echo "Installed pre-commit sensitive scanner hook at $HOOK"
