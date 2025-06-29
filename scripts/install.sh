BASE_DIR=$(pwd)
echo "BASE_DIR: $BASE_DIR"

python3 -m venv $BASE_DIR/venv
source $BASE_DIR/venv/bin/activate
# do inside which python is installed
echo "venv activated: $(which python)"

# show default place hf caches 
echo "hf caches: $HOME/.cache/huggingface"
ls -la $HOME/.cache/huggingface

# install huggingface-hub huggingface-cli transformers datasets
pip install huggingface-hub huggingface_hub[cli] transformers datasets

# Show the installed packages
pip freeze > $BASE_DIR/requirements.txt
cat $BASE_DIR/requirements.txt


# create in base dir a folder called "src"
mkdir -p $BASE_DIR/src

# Create Python script to download SmolVLA model
cat > $BASE_DIR/download_model.py << 'EOF'
from huggingface_hub import hf_hub_download

repo_id = "lerobot/smolvla_base"
local_dir = "smolvla_local"

# Download the main model files
hf_hub_download(repo_id=repo_id, filename="model.safetensors", local_dir=local_dir)
hf_hub_download(repo_id=repo_id, filename="config.json", local_dir=local_dir)
hf_hub_download(repo_id=repo_id, filename="preprocessor_config.json", local_dir=local_dir)

# Download tokenizer files
hf_hub_download(repo_id=repo_id, filename="tokenizer.json", local_dir=local_dir)
hf_hub_download(repo_id=repo_id, filename="merges.txt", local_dir=local_dir)
hf_hub_download(repo_id=repo_id, filename="special_tokens_map.json", local_dir=local_dir)
hf_hub_download(repo_id=repo_id, filename="tokenizer_config.json", local_dir=local_dir)

print(f"Model files downloaded to: {local_dir}")
EOF

# Execute the Python script to download the model
echo "Downloading SmolVLA model files..."
python $BASE_DIR/download_model.py

