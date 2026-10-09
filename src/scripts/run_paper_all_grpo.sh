
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ========== Dataset List with Sample Counts (question-only, English prompt) ==========
DATA_LIST=(
    "scienceqa_data_with_thinking2:12726"
    "ai2_arc_data_with_thinking2:1119"
    "geoqa_plus_data_with_thinking2:5010"
    "csqa_data_with_thinking2:9741"
    "logiqa2_data_with_thinking2:12567"
    "aokvqa_data_with_thinking2:17056"
    "mathqa_data_with_thinking2:29837"
    "aqua_rat_data_with_thinking2:97467"
)

# Calculate num_train_epochs based on sample count
get_epochs() {
    local count=$1
    if [ "${count}" -le 2000 ]; then
        echo 12
    elif [ "${count}" -le 10000 ]; then
        echo 3
    elif [ "${count}" -le 20000 ]; then
        echo 2
    else
        echo 1
    fi
}

total=${#DATA_LIST[@]}
echo "=========================================="
echo "  run_paper_all_grpo: ${total} datasets to train"
echo "=========================================="

for i in "${!DATA_LIST[@]}"; do
    data="${DATA_LIST[$i]%%:*}"
    count="${DATA_LIST[$i]##*:}"
    epochs=$(get_epochs "${count}")
    idx=$((i + 1))

    echo ""
    echo "=========================================="
    echo "  [${idx}/${total}] Starting dataset: ${data}"
    echo "  Samples: ${count}, num_train_epochs: ${epochs}"
    echo "  Time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=========================================="

    DATA_PATH="${data}" 
    NUM_TRAIN_EPOCHS="${epochs}" 
    . "${SCRIPT_DIR}/8b_paper_grpo.sh"

    echo ""
    echo "=========================================="
    echo "  [${idx}/${total}] Dataset ${data} training completed"
    echo "  Time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=========================================="
done

echo ""
echo "=========================================="
echo "  All ${total} datasets training completed!"
echo "  End time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="
