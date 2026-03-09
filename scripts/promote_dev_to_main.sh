#!/bin/bash
# =============================================================================
# promote_dev_to_main.sh — Merge dev into main with checkpoint migration
#
# Archives old LFS checkpoints from checkpoints/ → checkpoints_legacy/,
# merges the dev branch, then commits the new run_fresh.sh checkpoints.
#
# Prerequisites:
#   1. All dev work committed and pushed
#   2. run_fresh.sh has completed and checkpoints exist at FRESH_CKPT_DIR
#   3. Working tree clean on main
#
# Usage:
#   ./scripts/promote_dev_to_main.sh [/path/to/run_fresh/checkpoints]
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

FRESH_CKPT_DIR="${1:-/sietch_colab/data_share/cxt_scratch/checkpoints}"

echo "=== promote_dev_to_main.sh ==="
echo "Repo:          ${REPO_ROOT}"
echo "Fresh ckpts:   ${FRESH_CKPT_DIR}"
echo ""

# ---- Safety checks ----
CURRENT_BRANCH=$(git branch --show-current)
if [ "${CURRENT_BRANCH}" != "main" ]; then
    echo "ERROR: Must be on the main branch (currently on '${CURRENT_BRANCH}')."
    echo "  Run: git checkout main && git pull"
    exit 1
fi

if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "ERROR: Working tree is not clean. Commit or stash changes first."
    exit 1
fi

# ---- Step 1: Archive old checkpoints ----
echo "Step 1: Archiving old checkpoints → checkpoints_legacy/"

if [ -d "checkpoints" ] && [ "$(git ls-files checkpoints/ | wc -l)" -gt 0 ]; then
    mkdir -p checkpoints_legacy
    git mv checkpoints/* checkpoints_legacy/ 2>/dev/null || true

    for model_dir in checkpoints_legacy/*/; do
        [ -d "$model_dir" ] || continue
        model_name=$(basename "$model_dir")
        mkdir -p "checkpoints_legacy/${model_name}"
    done

    rmdir checkpoints 2>/dev/null || true
    git add checkpoints_legacy/
    git commit -m "Archive legacy checkpoints to checkpoints_legacy/"
    echo "  ✓ Old checkpoints archived"
else
    echo "  (no tracked checkpoints to archive)"
fi

# ---- Step 2: Merge dev ----
echo ""
echo "Step 2: Merging dev into main"
git merge dev -m "Merge dev into main: reproducible training pipeline"
echo "  ✓ dev merged"

# ---- Step 3: Copy fresh checkpoints ----
echo ""
echo "Step 3: Installing fresh checkpoints from run_fresh.sh output"
bash scripts/copy_checkpoints.sh "${FRESH_CKPT_DIR}"
git add checkpoints/
git commit -m "Add reproducible checkpoints from run_fresh.sh"
echo "  ✓ Fresh checkpoints committed"

# ---- Done ----
echo ""
echo "=== Done ==="
echo ""
echo "Checkpoint layout:"
echo "  checkpoints/          ← default (run_fresh.sh reproducible)"
echo "  checkpoints_legacy/   ← archived original checkpoints"
echo ""
echo "Verify with:  git log --oneline -5"
echo "Push with:    git push origin main"
