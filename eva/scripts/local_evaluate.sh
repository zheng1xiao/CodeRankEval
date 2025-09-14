# 定义多个模型路径
Models=(
    "/shd/zzr/xz/model/QwQ-32B"
)

for Model in "${Models[@]}"; do
    echo "Starting vLLM API server for model: $Model"

    # 启动 vLLM API 服务器（后台运行）
    python3 -m vllm.entrypoints.openai.api_server \
        --model "$Model" \
        --tensor-parallel-size 4 \
        --dtype auto \
        --gpu-memory-utilization 0.9 \
        --port 5000 \
        > vllm_server.log 2>&1 &


    # 记录 API 服务器的进程 ID
    VLLM_PID=$!

    # 等待 API 服务器启动
    sleep 100

    echo "Running evaluation for model: $Model"

    python evaluate.py --local-model "$Model" \
        --method pairwise \
        --data-path "../datasets/pairwise_data.json" \
        --temperature 0.7 \
        --max-tokens 32000

    # 关闭 vLLM 服务器
    kill $VLLM_PID
    sleep 30  # 等待进程完全结束
done