#!/usr/bin/env bash
# Put third_party/strata on the branch arc-moe: the pinned upstream commit plus patches/strata/*.patch, in order.
# Idempotent: when the branch already carries exactly these patches nothing happens. Local edits in the submodule
# are kept in a stash, never thrown away.
set -euo pipefail
. "$(dirname "$0")/lib.sh"
cd "$STRATA"
git cat-file -e "$STRATA_PIN^{commit}" 2>/dev/null || git fetch -q origin "$STRATA_PIN" || git fetch -q origin
mapfile -t patches < <(ls "$ROOT"/patches/strata/*.patch 2>/dev/null | sort)
want=$( (echo "$STRATA_PIN"; cat "${patches[@]}") | sha256sum | cut -c1-16)
have=$(git config --get arcmoe.applied 2>/dev/null || true)
if [ "$(git rev-parse --abbrev-ref HEAD 2>/dev/null)" = "$STRATA_BRANCH" ] && [ "$have" = "$want" ]; then
    ok "Strata ${STRATA_PIN:0:7} + ${#patches[@]} arc-moe patches (already applied)"
    exit 0
fi
if ! git diff --quiet || ! git diff --cached --quiet; then
    git stash push -q -m "arc-moe: local edits before re-applying the patches $(date +%F-%T)"
    warn "local edits in third_party/strata saved with git stash"
fi
git checkout -q -B "$STRATA_BRANCH" "$STRATA_PIN"
for p in "${patches[@]}"; do
    git -c user.name=arc-moe -c user.email=arc-moe@localhost am -q --3way "$p" \
        || { git am --abort || true; die "patch $(basename "$p") does not apply"; }
done
git config arcmoe.applied "$want"
ok "Strata ${STRATA_PIN:0:7} + ${#patches[@]} arc-moe patches"
