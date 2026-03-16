#!/bin/bash
# RAG system entry point
# Usage: bash run.sh <questions_txt_path> <predictions_out_path>

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python3 "${SCRIPT_DIR}/rag.py" "$1" "$2"
