#!/bin/bash
# Quick verification script for SGLang integration
# This script checks that all components are in place and ready for testing

set -e

echo "🔍 Verifying SGLang Integration for prime-rl (RFC #1615)"
echo "=========================================================="
echo ""

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Track results
PASSED=0
FAILED=0
WARNINGS=0

check_file() {
    if [ -f "$1" ]; then
        echo -e "${GREEN}✓${NC} $2"
        ((PASSED++))
    else
        echo -e "${RED}✗${NC} $2 - File not found: $1"
        ((FAILED++))
    fi
}

check_content() {
    if grep -q "$2" "$1" 2>/dev/null; then
        echo -e "${GREEN}✓${NC} $3"
        ((PASSED++))
    else
        echo -e "${RED}✗${NC} $3 - Pattern not found in $1"
        ((FAILED++))
    fi
}

warn() {
    echo -e "${YELLOW}⚠${NC} $1"
    ((WARNINGS++))
}

echo "📁 Checking File Structure..."
echo "------------------------------"
check_file "src/prime_rl/inference/server.py" "Router entrypoint exists"
check_file "src/prime_rl/inference/sglang/server.py" "SGLang server exists"
check_file "src/prime_rl/inference/sglang/__init__.py" "SGLang package init exists"
check_file "src/prime_rl/inference/sglang/worker/__init__.py" "SGLang worker module exists"
check_file "src/prime_rl/inference/config.py" "Inference config exists"
check_file "tests/integration/test_rl_sglang.py" "Integration tests exist"
check_file "docs/sglang_integration.md" "Documentation exists"
check_file "SGLANG_TESTING.md" "Testing guide exists"
check_file "SGLANG_INTEGRATION_SUMMARY.md" "Integration summary exists"
echo ""

echo "🔧 Checking Router Implementation..."
echo "-------------------------------------"
check_content "src/prime_rl/inference/server.py" "def main()" "Router main() function exists"
check_content "src/prime_rl/inference/server.py" "parse_argv" "Router uses parse_argv"
check_content "src/prime_rl/inference/server.py" "vllm_server" "Router imports vLLM server"
check_content "src/prime_rl/inference/server.py" "sglang_server" "Router imports SGLang server"
check_content "pyproject.toml" 'inference = "prime_rl.inference.server:main"' "Entry point configured"
echo ""

echo "🚀 Checking SGLang Server Implementation..."
echo "--------------------------------------------"
check_content "src/prime_rl/inference/sglang/server.py" "@app.get(\"/health\")" "Health endpoint exists"
check_content "src/prime_rl/inference/sglang/server.py" "@app.get(\"/v1/models\")" "Models endpoint exists"
check_content "src/prime_rl/inference/sglang/server.py" "@app.post(\"/v1/chat/completions\")" "Chat completions endpoint exists"
check_content "src/prime_rl/inference/sglang/server.py" "@app.post(\"/update_weights\")" "Update weights endpoint exists"
check_content "src/prime_rl/inference/sglang/server.py" "@app.post(\"/reload_weights\")" "Reload weights endpoint exists"
check_content "src/prime_rl/inference/sglang/server.py" "@app.post(\"/init_broadcaster\")" "NCCL broadcaster endpoint exists"
check_content "src/prime_rl/inference/sglang/server.py" "@app.post(\"/v1/chat/completions/tokens\")" "Pre-tokenized endpoint exists"
check_content "src/prime_rl/inference/sglang/server.py" "@app.post(\"/v1/load_lora_adapter\")" "Load LoRA endpoint exists"
check_content "src/prime_rl/inference/sglang/server.py" "@app.post(\"/v1/unload_lora_adapter\")" "Unload LoRA endpoint exists"
echo ""

echo "⚙️  Checking Configuration..."
echo "------------------------------"
check_content "src/prime_rl/inference/config.py" "backend.*Literal.*vllm.*sglang" "Backend field exists in config"
check_content "src/prime_rl/inference/config.py" "class InferenceConfig" "InferenceConfig class exists"
echo ""

echo "🧪 Checking Tests..."
echo "--------------------"
check_content "tests/integration/test_rl_sglang.py" "def test_no_error" "Basic error test exists"
check_content "tests/integration/test_rl_sglang.py" "def test_sglang_reward_goes_up" "Reward improvement test exists"
check_content "tests/integration/test_rl_sglang.py" "def test_sglang_reward_in_range" "Reward range test exists"
check_content "tests/integration/test_rl_sglang.py" "--inference.backend.*sglang" "Tests use SGLang backend"
echo ""

echo "📚 Checking Documentation..."
echo "-----------------------------"
check_content "docs/sglang_integration.md" "Architecture" "Architecture section exists"
check_content "docs/sglang_integration.md" "Quick Start" "Quick start section exists"
check_content "docs/sglang_integration.md" "API Endpoints" "API endpoints documented"
check_content "docs/sglang_integration.md" "Testing" "Testing section exists"
check_content "SGLANG_TESTING.md" "Test Plan" "Test plan exists"
check_content "SGLANG_TESTING.md" "Router Tests" "Router tests documented"
check_content "SGLANG_TESTING.md" "Integration Tests" "Integration tests documented"
echo ""

echo "🔌 Checking Dependencies..."
echo "---------------------------"
if grep -q "sglang\[all\]" pyproject.toml; then
    echo -e "${GREEN}✓${NC} SGLang dependency in pyproject.toml"
    ((PASSED++))
else
    echo -e "${RED}✗${NC} SGLang dependency missing from pyproject.toml"
    ((FAILED++))
fi

if [ -d "venv_sglang" ]; then
    echo -e "${GREEN}✓${NC} SGLang virtual environment exists"
    ((PASSED++))
else
    warn "SGLang virtual environment not found (run ./scripts/setup_env.sh)"
fi

if [ -d "venv_vllm" ]; then
    echo -e "${GREEN}✓${NC} vLLM virtual environment exists"
    ((PASSED++))
else
    warn "vLLM virtual environment not found (run ./scripts/setup_env.sh)"
fi
echo ""

echo "📊 Verification Summary"
echo "======================="
echo -e "${GREEN}Passed:${NC} $PASSED"
if [ $FAILED -gt 0 ]; then
    echo -e "${RED}Failed:${NC} $FAILED"
fi
if [ $WARNINGS -gt 0 ]; then
    echo -e "${YELLOW}Warnings:${NC} $WARNINGS"
fi
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}✅ All checks passed!${NC}"
    echo ""
    echo "The SGLang integration is complete and ready for testing."
    echo ""
    echo "Next steps:"
    echo "1. Set up GPU environment: ./scripts/setup_env.sh"
    echo "2. Run tests: See SGLANG_TESTING.md for detailed instructions"
    echo "3. Run benchmarks: ./scripts/run_benchmarks.sh"
    echo ""
    echo "For more information, see:"
    echo "- SGLANG_INTEGRATION_SUMMARY.md - Overview and status"
    echo "- docs/sglang_integration.md - User documentation"
    echo "- SGLANG_TESTING.md - Testing guide"
    exit 0
else
    echo -e "${RED}❌ Some checks failed!${NC}"
    echo ""
    echo "Please review the failed checks above and ensure all required files exist."
    echo "See SGLANG_INTEGRATION_SUMMARY.md for the complete file list."
    exit 1
fi
