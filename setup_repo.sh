#!/usr/bin/env bash
# setup_repo.sh
# Run this once to initialise the git repo and push to GitHub.
# Usage:
#   chmod +x setup_repo.sh
#   ./setup_repo.sh <github-username> <repo-name>
#
# Prerequisites:
#   - git installed
#   - GitHub CLI installed: https://cli.github.com/
#     OR a repo already created at github.com and its URL handy

set -euo pipefail

GITHUB_USER="${1:-your-username}"
REPO_NAME="${2:-vae-implication-testing}"
DESCRIPTION="Statistical implication testing using VAE latent spaces on Perturb-seq data"

echo "==> Initialising git repo..."
git init
git add .
git commit -m "Initial commit: VAE implication testing pipeline"

echo ""
echo "==> Choose how to create the GitHub remote:"
echo "    1) GitHub CLI (gh) — fully automated"
echo "    2) Manual — paste your repo URL"
read -rp "Choice [1/2]: " CHOICE

if [ "$CHOICE" = "1" ]; then
    # Requires: gh auth login
    gh repo create "$REPO_NAME" \
        --public \
        --description "$DESCRIPTION" \
        --source=. \
        --remote=origin \
        --push
    echo ""
    echo "==> Done! Repo live at: https://github.com/${GITHUB_USER}/${REPO_NAME}"
else
    echo ""
    echo "Create a new repo at: https://github.com/new"
    echo "  Name: $REPO_NAME"
    echo "  Description: $DESCRIPTION"
    echo "  Visibility: Public (or Private)"
    echo "  DO NOT initialise with README/.gitignore (we have our own)"
    echo ""
    read -rp "Paste the repo URL (e.g. https://github.com/user/repo.git): " REMOTE_URL
    git remote add origin "$REMOTE_URL"
    git branch -M main
    git push -u origin main
    echo ""
    echo "==> Done! Repo live at: $REMOTE_URL"
fi

echo ""
echo "==> Next steps for Jules:"
echo "    1. pip install -r requirements.txt"
echo "    2. python 01_download_and_prep.py"
echo "    3. python 02_train_vae.py"
echo "    4. python 03_implication_testing.py"
echo "    5. python 04_visualize.py"
