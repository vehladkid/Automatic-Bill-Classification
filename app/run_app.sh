#!/usr/bin/env bash
# Launch the HR Bill Classifier demo UI.  Usage: bash app/run_app.sh   -> open http://127.0.0.1:8501
cd "$(dirname "$0")/.." || exit 1
# models are cached locally; asking huggingface.co on every start can hang forever when the network (IPv6) is bad. Override with HF_HUB_OFFLINE=0 only to re-download.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
exec .venv/bin/streamlit run app/streamlit_app.py --server.address 127.0.0.1 --server.port 8501 --server.maxUploadSize 20 --browser.gatherUsageStats false --server.headless true "$@"
