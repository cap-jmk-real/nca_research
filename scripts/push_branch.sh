#!/usr/bin/env zsh
# Push current auto-research branch (run in Terminal.app after gh auth login).
# Usage: ./scripts/push_branch.sh [branch]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
BRANCH="${1:-$(git branch --show-current)}"

echo "=== push $BRANCH ==="
gh auth status -h github.com
curl -sS --max-time 8 -o /dev/null -w "api.github.com %{http_code}\n" https://api.github.com
git status -sb
git push -u origin "$BRANCH"
echo
echo "PR compare:"
echo "https://github.com/cap-jmk-real/nca_research/compare/main...${BRANCH}?expand=1"
