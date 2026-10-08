#!/bin/bash
# Fetch the arXiv source of DomainBed (Gulrajani & Lopez-Paz, arXiv:2007.01434v1), whose per-environment
# appendix tables feed the external ICC example (Appendix I). The parsed cells are already shipped in
# results/v2/external/domainbed_tables.csv; this script is only needed to re-run the parser:
#   python src/v2/analyze_v2.py domainbed
set -euo pipefail
cd "$(dirname "$0")/.."
D=results/v2/external/src_2007.01434
mkdir -p "$D/extracted"
curl -L -o "$D/eprint.tar.gz" https://arxiv.org/e-print/2007.01434v1
echo "c76db5455277454d1309e4fcb4fbd90bf56a70eaafce6ca07eb394e69e93081d  $D/eprint.tar.gz" | sha256sum -c
tar -xzf "$D/eprint.tar.gz" -C "$D/extracted"
echo "7451178f9e3e661bede8f8e9d14804fc292ec1e2c1eb8ec6b6fc7b2e991c17bf  $D/extracted/paper.tex" | sha256sum -c
