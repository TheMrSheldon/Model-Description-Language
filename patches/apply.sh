#!/bin/sh
# Applies the patches in directory $1 (in order) to the current directory, each unless it is applied already.
set -e
for p in "$1"/*.patch; do
	if patch -p1 -R --dry-run -s -f < "$p" > /dev/null 2>&1; then
		continue # (applied)
	fi
	patch -p1 -s -f < "$p"
done
