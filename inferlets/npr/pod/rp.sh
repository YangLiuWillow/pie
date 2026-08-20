# source this: exports RUNPOD_API_KEY / HF_TOKEN from ~/Documents/Liszt_ai/.env
export RUNPOD_API_KEY=$(grep '^RUNPOD_API_KEY=' ~/Documents/Liszt_ai/.env | cut -d= -f2- | tr -d '\r"'"'"' ')
export HF_TOKEN=$(grep '^HF_TOKEN=' ~/Documents/Liszt_ai/.env | cut -d= -f2- | tr -d '\r"'"'"' ')
export NPR_STATE=${NPR_STATE:-$HOME/.npr-pod}; mkdir -p "$NPR_STATE"
