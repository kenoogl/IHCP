# Julia版実行方法ガイド

## 🚀 Julia版IHCP-CGMソルバー実行方法

### 📋 必要環境

#### 1. Julia環境
- **Juliaバージョン**: 1.11以上
- **推奨**: Julia 1.11.0（最新LTS）

#### 2. システム要件
- **CPU**: 8コア以上推奨
- **メモリ**: 8GB以上
- **ストレージ**: 2GB以上の空き容量

### 🔧 環境設定

#### 1. Juliaインストール確認
```bash
julia --version
# Julia Version 1.11.0 (2024-10-07) などが表示されるか確認
```

#### 2. 必要パッケージのインストール
```bash
cd /Users/Daily/Development/IHCP/TrialClaude/julia_version

# パッケージインストール
julia -e "using Pkg; Pkg.add([\"NPZ\", \"IterativeSolvers\", \"SparseArrays\", \"LinearAlgebra\", \"Printf\", \"CSV\", \"DataFrames\", \"MAT\", \"Polynomials\"])"
```

#### 3. スレッド数設定（重要！）
```bash
# 8スレッドで実行（推奨）
export JULIA_NUM_THREADS=8

# または実行時指定
julia --threads 8
```

### 📁 ファイル構成確認

実行前に以下のファイルが存在することを確認：

```
julia_version/
├── IHCP_CGM_Julia.jl          # 基盤・熱物性値計算
├── dhcp_solver.jl             # 直接問題ソルバー
├── adjoint_solver.jl          # 隨伴問題ソルバー
├── cgm_solver.jl              # CGMアルゴリズム
├── metal_thermal_properties.csv # SUS304熱物性データ
└── T_measure_700um_1ms.npy    # 実測定データ（1.1GB）
```

### 🎮 実行方法

#### 1. 基本テスト実行
```bash
cd /Users/Daily/Development/IHCP/TrialClaude/julia_version

# 基本機能テスト
julia --threads 8 test_basic.jl

# 直接問題ソルバーテスト
julia --threads 8 test_dhcp.jl

# 隨伴問題ソルバーテスト
julia --threads 8 test_adjoint.jl

# CGMアルゴリズムテスト
julia --threads 8 test_cgm.jl
```

#### 2. 実データ統合テスト
```bash
# 実データを使用した統合テスト
JULIA_NUM_THREADS=8 julia real_data_test.jl
```

#### 3. 性能ベンチマーク実行
```bash
# 直接問題ソルバーベンチマーク
JULIA_NUM_THREADS=8 julia benchmark_comparison.jl
```

#### 4. インタラクティブ実行
```bash
# Julia REPL起動
julia --threads 8

# Julia REPLで実行
julia> include("real_data_test.jl")
```

### 📊 実行例と期待結果

#### 1. 基本テスト（test_basic.jl）
```
==============================
基本機能テスト開始
==============================

【テスト1】熱物性値計算
✓ 熱物性値計算: 正常

【テスト2】格子設定
✓ 格子設定: 正常

【最終結果】成功: 5/5 テスト
基本機能テスト: 成功 ✓
```

#### 2. 実データテスト（real_data_test.jl）
```
======================================================================
実データ統合テスト: Julia版 IHCP-CGM ソルバー
======================================================================

【テスト1】実データファイル読み込み
データ形状: (18143, 80, 100)
データ型: Float64
温度範囲: 419.58 - 588.06 K

【テスト4】小規模CGM実行（実データ）
CGM最適化開始...
CGM最適化完了。実行時間: 85.2秒
反復数: 20

🎉 Julia版IHCP-CGMソルバー: 実データテスト成功
```

#### 3. 性能ベンチマーク（benchmark_comparison.jl）
```
================================================================================
Python vs Julia 性能比較: 直接問題ソルバー（DHCP）
================================================================================

【Julia版実行】直接問題ソルバー測定
Run 1: 1.559秒
Run 2: 1.481秒
Run 3: 1.511秒

Julia版結果:
  平均実行時間: 1.559 ± 0.109 秒
  計算レート: 1026624 格子点×ステップ/秒
```

### 🐛 トラブルシューティング

#### 1. パッケージエラー
```bash
# パッケージの再インストール
julia -e "using Pkg; Pkg.rm([\"NPZ\", \"IterativeSolvers\"]); Pkg.add([\"NPZ\", \"IterativeSolvers\"])"

# 環境クリア
julia -e "using Pkg; Pkg.resolve()"
```

#### 2. メモリエラー
```bash
# Julia起動時のメモリ制限
julia --threads 8 --heap-size-hint=8G

# またはシステムメモリ確認
free -h  # Linux
vm_stat  # macOS
```

#### 3. スレッド数確認
```bash
julia --threads 8 -e "println(\"Threads: \", Threads.nthreads())"
# 出力: Threads: 8
```

### ⚡ 最適化設定

#### 1. 最高性能設定
```bash
# 環境変数設定
export JULIA_NUM_THREADS=8
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

# 最適化フラグ付きJulia実行
julia --threads 8 --optimize=3 --compile=yes real_data_test.jl
```

#### 2. バッチ実行用スクリプト
```bash
#!/bin/bash
# run_julia_batch.sh

export JULIA_NUM_THREADS=8
export OMP_NUM_THREADS=8

cd /Users/Daily/Development/IHCP/TrialClaude/julia_version

echo "=== Julia版IHCP-CGM実行開始 ==="
date

# 実データ統合テスト
julia --threads 8 real_data_test.jl

echo "=== 実行完了 ==="
date
```

実行:
```bash
chmod +x run_julia_batch.sh
./run_julia_batch.sh
```

### 📈 性能期待値

| テスト | 実行時間 | 備考 |
|---------|----------|------|
| **基本テスト** | 5-10秒 | 全テスト合計 |
| **小規模CGM** | 30-60秒 | 50フレーム×20反復 |
| **直接問題ベンチマーク** | 1.5秒 | 10時間ステップ |
| **フルスケールCGM** | 数時間 | 500フレーム×100反復 |

### 🎯 推奨実行フロー

#### 1. 初回セットアップ
```bash
# 1. 基本テスト
julia --threads 8 test_basic.jl

# 2. 個別機能テスト
julia --threads 8 test_dhcp.jl
julia --threads 8 test_adjoint.jl
julia --threads 8 test_cgm.jl
```

#### 2. 実用実行
```bash
# 3. 実データテスト
JULIA_NUM_THREADS=8 julia real_data_test.jl

# 4. 性能ベンチマーク
JULIA_NUM_THREADS=8 julia benchmark_comparison.jl
```

#### 3. 本格計算
```bash
# 5. フルスケール計算（時間要注意）
JULIA_NUM_THREADS=8 nohup julia real_data_full_scale.jl > output.log 2>&1 &
```

### ✨ 成功時の出力例

すべてが正しく動作している場合の最終出力：

```
🎉 Julia版IHCP-CGMソルバー: 実データテスト成功

主要成果:
✅ 実データ(1.1GB)の正常読み込み
✅ CGMアルゴリズムの正常動作
✅ 物理的に妥当な熱流束推定
✅ Python版との互換性確認
✅ 計算性能: 1026624 格子点×反復/秒
```

これで Julia版IHCP-CGMソルバーが正常に実行できます！