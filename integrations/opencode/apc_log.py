#!/usr/bin/env python3
"""Reassemble the serving inferlet's stderr from a pie server log.

The guest's `eprintln!` reaches the host one FORMAT ARGUMENT at a time, and
each fragment is logged as its own record interleaved with RPC chatter from
other threads. Grepping a fragment and taking the next line therefore reads
whichever thread logged next, not the rest of the sentence.

Filtering to `pie::inferlet:` records keeps only guest writes; concatenating
them in file order reconstructs exactly what the guest printed.
"""
import re
import sys

LINE = re.compile(r"pie::inferlet:\s?(.*)$")
ANSI = re.compile(r"\x1b\[[0-9;]*m")

path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/pie_apcA.log"
buf = []
with open(path, "rb") as f:
    for raw in f:
        m = LINE.search(ANSI.sub("", raw.decode("utf-8", "replace")).rstrip("\n"))
        if m:
            buf.append(m.group(1))
text = "".join(buf)

# Split on the marker, not on newlines: the trailing "\n" arrives inside the
# last fragment's record, where the log framing makes it indistinguishable from
# the record separator.
parts = text.split("[apc]")
head, entries = parts[0], parts[1:]
for i, e in enumerate(entries, 1):
    print(f"{i:>3}  [apc]{e.strip()}")
tail = [l for l in head.split("\n") if l.strip()]
if tail:
    print("\nother guest output:")
    for l in tail[-15:]:
        print(f"     {l}")
