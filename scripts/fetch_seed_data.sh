#!/usr/bin/env bash
# Download openly licensed seed corpora into data/seed/.
#
# - D&D 5.1 SRD: CC-BY-4.0, official Wizards of the Coast PDF.
# - OpenD6 core books: released under the OGL by West End Games; mirrored on
#   the Internet Archive. If a mirror URL rots, fetch manually and drop the
#   files into data/seed/opend6/.
#
# Failures are warnings, not errors: the pipeline runs with whatever subset
# is present.
set -u
cd "$(dirname "$0")/.."

mkdir -p data/seed/srd_5.1 data/seed/opend6

fetch() {
    local url=$1 out=$2
    if [ -s "$out" ]; then
        echo "already have $out"
        return 0
    fi
    echo "fetching $url"
    if ! curl -fL --retry 3 --connect-timeout 15 -o "$out" "$url"; then
        echo "WARN: failed to fetch $url — fetch manually into $out" >&2
        rm -f "$out"
    fi
}

fetch "https://media.wizards.com/2023/downloads/dnd/SRD_CC_v5.1.pdf" \
    "data/seed/srd_5.1/SRD_CC_v5.1.pdf"

fetch "https://archive.org/download/d6-fantasy/D6%20Fantasy.pdf" \
    "data/seed/opend6/D6_Fantasy.pdf"
fetch "https://archive.org/download/d6-adventure/D6%20Adventure.pdf" \
    "data/seed/opend6/D6_Adventure.pdf"

echo "seed data:"
find data/seed -type f -exec ls -lh {} \;
