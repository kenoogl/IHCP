#!/bin/bash
# Julia版IHCP-CGMフルスケール実行スクリプト
# オリジナルPython版と同等の完全計算

echo "=========================================================================="
echo "Julia版 IHCP-CGM フルスケール実行スクリプト"
echo "=========================================================================="

# 環境変数設定
export JULIA_NUM_THREADS=8
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

echo "実行設定:"
echo "  Juliaスレッド数: $JULIA_NUM_THREADS"
echo "  OpenMPスレッド数: $OMP_NUM_THREADS"
echo "  作業ディレクトリ: $(pwd)"
echo "  実行開始時刻: $(date)"

# システムリソース確認
echo ""
echo "システムリソース確認:"
echo "  CPU情報:"
if command -v nproc &> /dev/null; then
    echo "    論理プロセッサ数: $(nproc)"
elif command -v sysctl &> /dev/null; then
    echo "    論理プロセッサ数: $(sysctl -n hw.logicalcpu)"
fi

echo "  メモリ情報:"
if command -v free &> /dev/null; then
    free -h | grep -E "Mem:|Swap:"
elif command -v vm_stat &> /dev/null; then
    echo "    $(vm_stat | head -4)"
fi

echo "  ディスク容量:"
df -h . | tail -1

# 必要ファイル確認
echo ""
echo "必要ファイル確認:"
required_files=(
    "julia_full_scale_execution.jl"
    "cgm_solver.jl"
    "adjoint_solver.jl"
    "dhcp_solver.jl"
    "IHCP_CGM_Julia.jl"
    "metal_thermal_properties.csv"
    "T_measure_700um_1ms.npy"
)

all_files_present=true
for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✅ $file"
    else
        echo "  ❌ $file (見つかりません)"
        all_files_present=false
    fi
done

if [ "$all_files_present" = false ]; then
    echo ""
    echo "❌ 必要なファイルが不足しています。実行を中止します。"
    exit 1
fi

# Juliaバージョン確認
echo ""
echo "Julia環境確認:"
if command -v julia &> /dev/null; then
    julia_version=$(julia --version)
    echo "  $julia_version"

    # スレッド数確認
    thread_test=$(julia --threads $JULIA_NUM_THREADS -e "println(\"Threads: \", Threads.nthreads())")
    echo "  $thread_test"
else
    echo "  ❌ Juliaがインストールされていません"
    exit 1
fi

# ログファイル設定
log_file="julia_fullscale_$(date +%Y%m%d_%H%M%S).log"
echo ""
echo "実行ログファイル: $log_file"

# 実行確認
echo ""
echo "⚠️  注意: フルスケール実行は数時間から十数時間かかる場合があります"
echo "    実データサイズ: ~1.1GB"
echo "    推定実行時間: 2-8時間（ハードウェアによる）"
echo "    結果ファイル: ~100MB"
echo ""

# 確認プロンプト（バッチ実行時はスキップ）
if [ -t 0 ]; then
    read -p "実行を続行しますか？ (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "実行を中止しました"
        exit 0
    fi
fi

# バックグラウンド実行の提案
echo ""
echo "実行方法を選択してください:"
echo "  1) フォアグラウンド実行（ターミナル占有）"
echo "  2) バックグラウンド実行（推奨）"
echo ""

if [ -t 0 ]; then
    read -p "選択 (1/2, デフォルト: 2): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[1]$ ]]; then
        execution_mode="foreground"
    else
        execution_mode="background"
    fi
else
    execution_mode="background"
fi

echo ""
echo "=========================================================================="
echo "Julia版フルスケール計算実行開始"
echo "=========================================================================="

# 実行開始
start_time=$(date +%s)

if [ "$execution_mode" = "background" ]; then
    echo "バックグラウンド実行開始..."
    echo "  ログファイル: $log_file"
    echo "  進捗確認: tail -f $log_file"
    echo "  プロセス確認: ps aux | grep julia"
    echo ""

    # バックグラウンド実行
    nohup julia --threads $JULIA_NUM_THREADS julia_full_scale_execution.jl > "$log_file" 2>&1 &
    julia_pid=$!

    echo "Juliaプロセス開始: PID=$julia_pid"
    echo ""
    echo "実行状況の確認方法:"
    echo "  tail -f $log_file                    # ログ監視"
    echo "  grep -E 'ウィンドウ|進捗|完了' $log_file  # 進捗確認"
    echo "  kill $julia_pid                     # 実行停止"
    echo ""
    echo "バックグラウンド実行を開始しました"

else
    echo "フォアグラウンド実行開始..."
    echo "  Ctrl+C で中断可能"
    echo ""

    # フォアグラウンド実行
    julia --threads $JULIA_NUM_THREADS julia_full_scale_execution.jl 2>&1 | tee "$log_file"
fi

# 実行完了確認（フォアグラウンドの場合のみ）
if [ "$execution_mode" = "foreground" ]; then
    end_time=$(date +%s)
    elapsed=$((end_time - start_time))

    echo ""
    echo "=========================================================================="
    echo "実行完了"
    echo "=========================================================================="
    echo "  総実行時間: $(($elapsed / 3600))時間$(($elapsed % 3600 / 60))分$(($elapsed % 60))秒"
    echo "  実行ログ: $log_file"
    echo "  完了時刻: $(date)"

    # 結果ファイル確認
    result_files=$(ls julia_ihcp_results_*.npz 2>/dev/null)
    if [ -n "$result_files" ]; then
        echo ""
        echo "生成された結果ファイル:"
        for file in $result_files; do
            size=$(du -h "$file" | cut -f1)
            echo "  📁 $file ($size)"
        done
    fi

    echo ""
    echo "✅ Julia版フルスケール実行が完了しました"
fi

echo ""
echo "=========================================================================="