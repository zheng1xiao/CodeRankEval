# 定义多个模型路径
Models=(
    "DeepSeek-R1"
)

for Model in "${Models[@]}"; do

    echo "Running evaluation for model: $Model"


    python evaluate.py --api-model "$Model" \
        --method pairwise \
        --data-path "../datasets/pairwise_data.json" \
        --temperature 0.7 \

    sleep 10
done