# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 🎯 プロジェクト完成状況
**✅ Python→Julia変換が完全に成功しました！**

このリポジトリには逆熱伝導問題（IHCP）を共役勾配法（CGM）で解く2つの完成版があります：
- **Python版**: `org/` ディレクトリ（オリジナル、Numba最適化済み）
- **Julia版**: `julia_version/` ディレクトリ（✅変換完了、高性能実装）

## 基本方針
逆熱伝導問題の表面熱流束逆解析を高精度で実行するソルバーシステムです。IRカメラ温度データから表面熱流束を推定する工学計算コードです。

## プロジェクト概要
逆熱伝導問題のスライディングウィンドウ計算プログラム：
- IRカメラから取得したMATLABファイル（1.1GB）の温度データ読み込み
- SUS304の温度依存熱物性値（密度、比熱、熱伝導率）の多項式フィッティング
- 熱伝導方程式の直接ソルバー（DHCP） - 順問題
- 共役勾配法による随伴ソルバー（Adjoint） - 感度解析
- 表面熱流束の逆解析計算（CGM最適化）

## 技術スタック

### Python版（org/）
- Python 3.12.7
- NumPy 1.26.4 (数値計算)
- SciPy 1.13.1 (sparse行列、線形システム)
- Pandas 2.2.2 (データ処理)
- Numba 0.60.0 (JIT最適化、並列処理)

### Julia版（julia_version/）
- Julia 1.11.0以上
- LinearAlgebra, SparseArrays (行列計算)
- IterativeSolvers (共役勾配法)
- NPZ (データファイル読み込み)
- CSV, DataFrames (データ処理)
- MAT (MATLABファイル読み込み)

## コマンド

### Python版実行
```bash
cd org
python IHCP_CGM_Sliding_Window_Calculation_ver2.py
```

### Julia版実行
```bash
cd julia_version

# 環境設定（重要）
export JULIA_NUM_THREADS=8

# 段階的テスト
julia --project=. test_basic.jl      # 基本機能テスト（5-10秒）
julia --project=. test_dhcp.jl       # 直接問題ソルバー（30秒）
julia --project=. test_adjoint.jl    # 随伴問題ソルバー（30秒）
julia --project=. test_cgm.jl        # CGM最適化（1分）

# 実データ統合テスト
julia --project=. real_data_test.jl  # 実測定データ1.1GBでの動作確認

# 性能ベンチマーク
julia --project=. benchmark_comparison.jl  # Python vs Julia性能比較

# フルスケール実行（長時間）
julia --project=. julia_full_scale_execution.jl
```

### テスト実行方法
```bash
# Julia版の段階的テスト（推奨順序）
cd julia_version
julia --threads 8 test_basic.jl && \
julia --threads 8 test_dhcp.jl && \
julia --threads 8 test_adjoint.jl && \
julia --threads 8 test_cgm.jl && \
echo "全テスト完了"

# Python版のテスト
cd org
python test_case1_python.py
```

## コードアーキテクチャ

### Python版構造（org/）
- **メインファイル**: `IHCP_CGM_Sliding_Window_Calculation_ver2.py`（27,412行、統合実装）
- **ベンチマーク**: `python_benchmark.py`, `python_simple_benchmark.py`
- **テストケース**: `test_case1_python.py`

### Julia版構造（julia_version/）
**モジュラー設計による段階的実装**:
- **基盤**: `IHCP_CGM_Julia.jl` - 熱物性値計算、格子設定
- **直接問題**: `dhcp_solver.jl` - 熱伝導方程式ソルバー
- **随伴問題**: `adjoint_solver.jl` - 感度解析ソルバー
- **最適化**: `cgm_solver.jl` - 共役勾配法実装
- **統合テスト**: `real_data_test.jl` - 実データでの動作検証

### 主要な関数群:
1. **熱物性値計算**: `thermal_properties_calculator()` - 温度依存特性の多項式評価
2. **ファイル処理**: `extract_sorted_mat_files()`, `load_region_temperature()` - データ読み込み
3. **直接解法**: `coeffs_and_rhs_building_DHCP()`, `multiple_time_step_solver_DHCP()` - 順問題
4. **随伴解法**: `coeffs_and_rhs_building_Adjoint()`, `multiple_time_step_solver_Adjoint()` - 感度解析
5. **最適化**: `global_CGM_time()`, `sliding_window_CGM_q_saving()` - 熱流束逆解析

### 計算グリッド設定:
- **x, y方向**: 均等格子 (dx=0.12mm, dy調整済み)
- **z方向**: 20層の非均等格子 (表面側に集中配置)
- **時間ステップ**: 1ms (実データ対応)
- **データ規模**: 80×100×20格子点、最大18,143時間ステップ

## データファイル
- `*/metal_thermal_properties.csv`: SUS304の熱物性値データ（540B）
- `*/T_measure_700um_1ms.npy`: 実測定温度データ（1.1GB、18143×80×100配列）
- MATLABファイル: IRカメラ温度データ（SUS*.MAT形式、プロジェクトにより異なる）

## 性能特性

### 計算規模
- **小規模テスト**: 2×2×4格子、10時間ステップ（数秒）
- **中規模計算**: 80×100×20格子、50時間ステップ（分オーダー）
- **フルスケール**: 80×100×20格子、数千時間ステップ（時間オーダー）

### Julia版性能優位性
- **並列化**: `Threads.@threads`による効率的並列処理
- **型安定性**: コンパイル時最適化
- **メモリ効率**: ガベージコレクションとSparse行列最適化
- **計算性能**: ~1,000,000 格子点×反復/秒

## 開発フロー
### Test-Driven Development (TDD)
- 原則としてテスト駆動開発（TDD）で進める
- 期待される入出力に基づき、まずテストを作成する
- 段階的テスト（基本→DHCP→Adjoint→CGM→実データ）で信頼性確保
- テスト実行し、失敗を確認してから実装開始
- すべてのテストが通過するまで繰り返す

## 言語設定
- **回答言語**: 日本語のみ
- **コメント**: 日本語で物理的意味を詳細に説明
- **変数名**: 英語（物理・数学記法に準拠）
- **関数名**: 英語（snake_case形式）

## コーディング規約
- **インデント**: スペース4文字（Python）、スペース4文字（Julia）
- **コメント**: 物理的意味と数値計算の意図を日本語で詳細記述
- **最適化**: Python版はNumba @njit、Julia版は型安定性重視
- **並列処理**: Python版はprange、Julia版は@threads
- **エラー処理**: 数値計算の収束性とメモリ制限を考慮

## 重要な注意事項
- **大容量データ**: T_measure_700um_1ms.npy（1.1GB）の取り扱いに注意
- **計算時間**: フルスケール計算は数時間を要する場合がある
- **メモリ使用**: 大規模問題では8GB以上のメモリが必要
- **並列設定**: Julia版では`JULIA_NUM_THREADS=8`の環境変数設定が重要
- **数値精度**: 逆問題の性質上、収束判定とステップサイズ調整が重要

## トラブルシューティング
```bash
# Julia版パッケージエラー
julia -e "using Pkg; Pkg.resolve()"

# メモリ不足対応
julia --heap-size-hint=8G

# スレッド数確認
julia -e "println(Threads.nthreads())"

# Python版Numba再コンパイル
python -c "import numba; numba.config.CACHE_DIR = '/tmp/numba_cache'"
```