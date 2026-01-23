#!/bin/bash
# Fixed Environment Setup for PRIME-RL (Forcing Python 3.12)

echo "📦 [1/5] Installing Python 3.12 and dependencies..."
sudo apt update && sudo apt install -y software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt install -y python3.12 python3.12-venv python3.12-dev

echo "🐍 [2/5] Creating virtual environments (FORCING 3.12)..."
rm -rf venv_vllm venv_sglang
python3.12 -m venv venv_vllm
python3.12 -m venv venv_sglang

echo "🚀 [3/5] Building vLLM Environment..."
source venv_vllm/bin/activate
pip install -U pip uv
uv pip install -e ".[all]"
uv pip install git+https://github.com/samsja/dion.git@main
uv pip install git+https://github.com/pytorch/torchtitan.git@a1fdd7e
uv pip install git+https://github.com/PrimeIntellect-ai/verifiers.git@eb3bae36c93b2f574e13a39da2908609d4ec4b2f
sed -i 's/"sglang\[all\]>=0.4.4",//' pyproject.toml
uv pip install -e ".[all]"
deactivate

echo "🧹 [4/5] Nuclear GPU Cleanup before SGLang build..."
pkill -9 -f vllm
pkill -9 -f sglang
pkill -9 -f python
nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9 2>/dev/null
sleep 5

echo "🔥 [5/5] Building SGLang Environment..."
source venv_sglang/bin/activate
pip install -U pip uv
uv pip install "sglang[all]" openai httpx
deactivate

echo "✅ Setup complete. Run ./scripts/run_benchmarks.sh to verify performance."
