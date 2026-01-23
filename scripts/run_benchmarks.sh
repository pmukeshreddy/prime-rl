#!/bin/bash
# Native SGLang vs vLLM Benchmark Orchestrator

MODEL="Qwen/Qwen2.5-7B-Instruct"

echo "🧹 [Step 1/5] Nuclear Cleanup..."
pkill -9 -f vllm
pkill -9 -f sglang
pkill -9 -f python
sleep 10
nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9

echo -e "\n🚀 [Step 2/5] Running vLLM Baseline..."
source venv_vllm/bin/activate
# Use TP=2 for fair comparison (both GPUs, same as SGLang DP=2)
python3 -m vllm.entrypoints.openai.api_server \
  --model $MODEL --port 8000 --enable-prefix-caching --tensor-parallel-size 2 > /tmp/vllm.log 2>&1 &
sleep 120
python scripts/prefix_benchmark.py vllm $MODEL
pkill -9 -f vllm
deactivate

echo -e "\n☢️  [Step 3/5] Clearing VRAM for SGLang..."
sleep 20
nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9

echo -e "\n🚀 [Step 4/5] Running SGLang Challenge (DP=2)..."
source venv_sglang/bin/activate
python3 -m sglang.launch_server \
  --model-path $MODEL --port 8000 --dp 2 > /tmp/sglang.log 2>&1 &
sleep 120
python scripts/prefix_benchmark.py sglang $MODEL
deactivate

echo -e "\n📊 [Step 5/5] Final Performance Comparison"
echo "=============================================="
cat prefix_results_vllm.json
echo "----------------------------------------------"
cat prefix_results_sglang.json
echo "=============================================="
