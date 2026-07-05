#!/usr/bin/env bash
# Download openly licensed seed corpora into data/seed/.
#
# - D&D 5.1 SRD: CC-BY-4.0, official Wizards of the Coast PDF.
# - OpenD6 books: released under the OGL by West End Games in 2009; hosted
#   OCR'd by the OGC Library (ogc.rpglibrary.org).
#
# Failures are warnings, not errors: the pipeline runs with whatever subset
# is present.
set -u
cd "$(dirname "$0")/.."

mkdir -p data/seed/srd_5.1 data/seed/opend6/{rules,bestiary,setting}

fetch() {
    local url=$1 out=$2
    if [ -s "$out" ]; then
        echo "already have $out"
        return 0
    fi
    echo "fetching $url"
    if ! curl -fsSL -A "Mozilla/5.0 (X11; Linux x86_64)" --retry 3 \
            --connect-timeout 15 -o "$out" "$url"; then
        echo "WARN: failed to fetch $url — fetch manually into $out" >&2
        rm -f "$out"
    fi
}

fetch "https://media.wizards.com/2023/downloads/dnd/SRD_CC_v5.1.pdf" \
    "data/seed/srd_5.1/SRD_CC_v5.1.pdf"

OGC=https://ogc.rpglibrary.org/images
fetch "$OGC/7/72/D6_Fantasy_v1.3_weg51013OGL.pdf" data/seed/opend6/rules/D6_Fantasy.pdf
fetch "$OGC/1/1b/D6_Adventure_v2.0_weg51011OGL.pdf" data/seed/opend6/rules/D6_Adventure.pdf
fetch "$OGC/0/0b/D6_Magic_weg51024OGL.pdf" data/seed/opend6/rules/D6_Magic.pdf
fetch "$OGC/b/bf/D6_System_Book_weg51005eOGL.pdf" data/seed/opend6/rules/D6_System_Book.pdf
fetch "$OGC/0/09/D6_Adventure_Creatures_weg51021eOGL.pdf" \
    data/seed/opend6/bestiary/D6_Adventure_Creatures.pdf
fetch "$OGC/9/92/D6_Fantasy_Locations_v1.1_weg51020OGL.pdf" \
    data/seed/opend6/setting/D6_Fantasy_Locations.pdf
fetch "$OGC/a/a8/D6_Adventure_Locations_v1.1_weg51016eOGL.pdf" \
    data/seed/opend6/setting/D6_Adventure_Locations.pdf

echo "seed data:"
find data/seed -type f -exec ls -lh {} \;
