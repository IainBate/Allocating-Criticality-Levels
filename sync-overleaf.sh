#!/bin/bash

# Sync script for pushing local .tex files to Overleaf
# Usage: ./sync-overleaf.sh [commit-message]

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OVERLEAF_REMOTE="overleaf"

# Find all .tex files
TEX_FILES=$(find "$PROJECT_DIR" -name "*.tex" -type f 2>/dev/null)

if [ -z "$TEX_FILES" ]; then
    echo "No .tex files found in $PROJECT_DIR"
    exit 1
fi

echo "Found .tex files:"
echo "$TEX_FILES" | while read -r file; do
    echo "  - $(basename "$file")"
done

# Ask for confirmation
echo ""
read -p "Push these files to Overleaf? [y/N] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    exit 0
fi

# Stage all .tex files
cd "$PROJECT_DIR"
git add $(find "$PROJECT_DIR" -name "*.tex" -type f)

# Commit with message or default
COMMIT_MSG="${1:-Update LaTeX files}"
git commit -m "$COMMIT_MSG"

# Push to Overleaf
echo "Pushing to Overleaf..."
git push "$OVERLEAF_REMOTE" main

echo ""
echo "Sync complete! Your changes are now on Overleaf."
echo "View your project at: https://www.overleaf.com/project/6a856d6b86a8f0981d8372bb"
