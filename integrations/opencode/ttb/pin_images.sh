#!/usr/bin/env bash
# Record the DIGEST of every grader image actually present, and verify it later.
#
# test-time-bench pins its scorer by digest
# (`sslee0cs/ttb-swe-bench-scorer@sha256:ca4bd811…`). This repo pulled
# `swebench/sweb.eval.x86_64.*:latest`, which is a moving target: a re-run next
# month can grade against a different image and nothing in the artifact would
# say so. `latest` is fine for exploring and wrong for a recorded number.
#
#   ./pin_images.sh write   > images/digests.json   (after pulling)
#   ./pin_images.sh verify                          (before driving)
#   ./pin_images.sh pull [substr]                   (before grading)
#
# MISSING and DRIFTED are NOT the same failure:
#
#   * DRIFT  — an image is present and its digest differs from the pin. This is
#              an integrity failure: a number graded now is not comparable to
#              one graded before. `verify` exits non-zero.
#   * MISSING — the image is simply absent. Recoverable, because a pin is a
#              DIGEST: `pull` fetches `repo@sha256:...`, which is byte-for-byte
#              the pinned image or nothing at all. `verify` reports and passes.
#
# A DEAD DAEMON IS NEITHER, and conflating it with "missing" is how this script
# briefly reported all 31 images gone and exited 0: `run.sh` stops the colima VM
# to free memory for the model, so a `verify` run afterwards asks a daemon that
# is not there, every `docker inspect` fails, and absence looks unanimous. It
# was misread as "the grader deleted them"; the images were on the VM's disk the
# whole time. So the daemon is now probed first and an unreachable one is fatal:
# a check that cannot see anything must not report that everything is fine.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PIN="${TTB_PIN_FILE:-$HERE/images/digests.json}"
MODE="${1:-write}"

command -v docker >/dev/null || { echo "FATAL: no docker on PATH" >&2; exit 1; }
# Reachability, not just presence of the binary.
if ! docker info >/dev/null 2>&1; then
    echo "FATAL: the docker daemon is not reachable — cannot verify or pull." >&2
    echo "  colima start --cpu 6 --memory 14 --disk 80 --vm-type vz --vz-rosetta" >&2
    echo "(If run.sh stopped the VM to free memory for the model, that is expected:" >&2
    echo " verify pins BEFORE the drive, and pull images before grading.)" >&2
    exit 1
fi

# RepoDigests, not Id: the digest is what `docker pull` resolves and what a
# different machine can reproduce. Id is local to this daemon's storage.
list() {
    docker images --format '{{.Repository}}:{{.Tag}}' \
        | grep '^swebench/sweb\.eval\.' | sort -u \
        | while read -r img; do
            d=$(docker inspect --format '{{index .RepoDigests 0}}' "$img" 2>/dev/null || true)
            [ -n "$d" ] && printf '%s\t%s\n' "$img" "${d#*@}"
        done
}

case "$MODE" in
write)
    mkdir -p "$(dirname "$PIN")"
    {
        echo '{'
        echo '  "note": "Grader image digests as pulled. Regenerate only when deliberately re-pinning; a silent change here changes what a recorded number means.",'
        echo '  "images": {'
        first=1
        while IFS=$'\t' read -r img dig; do
            [ -z "$img" ] && continue
            [ $first -eq 0 ] && echo ','
            printf '    "%s": "%s"' "$img" "$dig"
            first=0
        done < <(list)
        echo
        echo '  }'
        echo '}'
    }
    ;;
verify)
    [ -f "$PIN" ] || { echo "FATAL: no pin file at $PIN — run 'write' first" >&2; exit 1; }
    fail=0; missing=0; present=0
    while IFS=$'\t' read -r img want; do
        have=$(docker inspect --format '{{index .RepoDigests 0}}' "$img" 2>/dev/null | sed 's/.*@//' || true)
        if [ -z "$have" ]; then
            missing=$((missing+1))
        elif [ "$have" != "$want" ]; then
            echo "DRIFT   $img" >&2
            echo "        pinned $want" >&2
            echo "        local  $have" >&2
            fail=1
        else
            present=$((present+1))
        fi
    done < <(python3 -c "
import json,sys
d=json.load(open('$PIN'))['images']
for k,v in d.items(): print(f'{k}\t{v}')
")
    if [ $fail -eq 0 ]; then
        if [ "$missing" -gt 0 ]; then
            echo "$present pinned images present and matching; $missing absent"
            echo "(absent is expected — the grader deletes what it evaluates."
            echo " run './pin_images.sh pull' before scoring; it fetches by DIGEST.)"
        else
            echo "all pinned grader images match"
        fi
    else
        echo "grader images DRIFTED from the pin — a run graded now is not comparable" >&2
        exit 1
    fi
    ;;
pull)
    # By digest, not by tag: `repo@sha256:...` either resolves to the pinned
    # image or fails. A tag pull could quietly bring back something newer.
    [ -f "$PIN" ] || { echo "FATAL: no pin file at $PIN" >&2; exit 1; }
    filter="${2:-}"
    n=0; bad=0
    while IFS=$'\t' read -r img want; do
        [ -n "$filter" ] && case "$img" in *"$filter"*) ;; *) continue ;; esac
        repo="${img%%:*}"
        if docker image inspect "$img" >/dev/null 2>&1; then continue; fi
        if docker pull --platform linux/amd64 "$repo@$want" >/dev/null 2>&1; then
            # Restore the tag the harness looks up by name.
            docker tag "$repo@$want" "$img" >/dev/null 2>&1 || true
            n=$((n+1))
        else
            echo "FAILED to pull $repo@$want" >&2; bad=$((bad+1))
        fi
    done < <(python3 -c "
import json,sys
d=json.load(open('$PIN'))['images']
for k,v in d.items(): print(f'{k}\t{v}')
")
    echo "pulled $n pinned image(s) by digest, $bad failed"
    [ $bad -eq 0 ]
    ;;
*)
    echo "usage: pin_images.sh [write|verify]" >&2; exit 2 ;;
esac
