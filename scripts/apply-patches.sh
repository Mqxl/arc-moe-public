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
# A file whose content comes out the same keeps its old timestamp, so re-applying a patch series that changed only
# setup_intel.py does not make the next ./arc build recompile the engine (a full AOT link is 15-60 min).
snap=$(mktemp)
trap 'rm -f "$snap"' EXIT
python3 - "$snap" <<'PY' || true
import hashlib, json, os, subprocess, sys
files = subprocess.run(["git", "ls-files", "-z", "sycl", "include", "src", "cmake", "CMakeLists.txt"],
                       capture_output=True).stdout.split(b"\0")
out = {}
for f in filter(None, files):
    try:
        st = os.stat(f)
        out[f.decode()] = [hashlib.sha1(open(f, "rb").read()).hexdigest(), st.st_atime_ns, st.st_mtime_ns]
    except OSError:
        pass
json.dump(out, open(sys.argv[1], "w"))
PY
git checkout -q -B "$STRATA_BRANCH" "$STRATA_PIN"
for p in "${patches[@]}"; do
    git -c user.name=arc-moe -c user.email=arc-moe@localhost am -q --3way "$p" \
        || { git am --abort || true; die "patch $(basename "$p") does not apply"; }
done
python3 - "$snap" <<'PY' || true
import hashlib, json, os, sys
kept = 0
for f, (h, at, mt) in json.load(open(sys.argv[1])).items():
    try:
        if hashlib.sha1(open(f, "rb").read()).hexdigest() == h:
            os.utime(f, ns=(at, mt)); kept += 1
    except OSError:
        pass
PY
git config arcmoe.applied "$want"
ok "Strata ${STRATA_PIN:0:7} + ${#patches[@]} arc-moe patches"
