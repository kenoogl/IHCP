# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 🎯 プロジェクト完成状況
**✅ Python→Julia変換が完全に成功しました！**

このリポジトリには逆熱伝導問題（IHCP）を共役勾配法（CGM）で解く2つの完成版があります：
- **Python版**: `org/` ディレクトリ（オリジナル）
- **Julia版**: `julia_version/` ディレクトリ（✅変換完了）

## 基本方針
逆熱伝導問題の表面熱流束逆解析を高精度で実行するソルバーシステムです。

## プロジェクト概要
逆熱伝導問題のスライディングウィンドウ計算プログラム：
- IRカメラから取得したMATLABファイルの温度データ読み込み
- SUS304の熱物性値（密度、比熱、熱伝導率）の多項式フィッティング
- 熱伝導方程式の直接ソルバー（DHCP）
- 共役勾配法による随伴ソルバー（Adjoint）
- 表面熱流束の逆解析計算

## 技術スタック
- Python 3.12.7
- NumPy (数値計算)
- SciPy (sparse行列、最適化)
- Pandas (データ処理)
- Numba (高速化、並列処理)

## コードアーキテクチャ
メインファイル: `org/IHCP_CGM_Sliding_Window_Calculation_ver2.py`

### 主要な関数群:
1. **熱物性値計算**: `thermal_properties_calculator()` - 温度依存熱物性値の計算
2. **ファイル処理**: `extract_sorted_mat_files()`, `load_region_temperature()` - MATLABファイル読み込み
3. **直接解法**: `coeffs_and_rhs_building_DHCP()`, `multiple_time_step_solver_DHCP()` - 熱伝導方程式の直接解法
4. **随伴解法**: `coeffs_and_rhs_building_Adjoint()`, `multiple_time_step_solver_Adjoint()` - CGM用随伴方程式
5. **最適化**: `global_CGM_time()`, `sliding_window_CGM_q_saving()` - 共役勾配法による逆解析

### 計算グリッド:
- x, y方向: 均等格子 (dx=0.12mm, dy調整済み)
- z方向: 20層の非均等格子 (表面側に集中)
- 時間ステップ: 1ms

## 実行方法

### Python版
```bash
cd org
python IHCP_CGM_Sliding_Window_Calculation_ver2.py
```

### Julia版 ✅
```bash
cd julia_version
julia --project=. real_data_test.jl  # 実データ統合テスト
julia --project=. test_cgm.jl        # CGMアルゴリズムテスト
julia --project=. test_dhcp.jl       # 直接問題ソルバーテスト
julia --project=. test_adjoint.jl    # 隨伴問題ソルバーテスト
```

## データファイル
- `org/metal_thermal_properties.csv`: SUS304の熱物性値データ
- `org/T_measure_700um_1ms.npy`: 測定温度データ（大容量）
- MATLABファイル: IRカメラ温度データ（SUS*.MAT形式）

## 言語設定
- **回答言語**: 日本語のみ
- **コメント**: 日本語で詳細に記述
- **変数名**: 英語（業界標準に準拠）
- **関数名**: 英語（snake_case形式）

## コーディング規約
- インデント: スペース4文字（Pythonスタンダード）
- コメント: 処理の意図を日本語で詳細に説明
- Numba最適化: 計算集約的関数には@njitデコレータ使用
- 並列処理: prange使用で性能向上

## 開発フロー
### Test-Driven Development (TDD)
- 原則としてテスト駆動開発（TDD）で進める
- 期待される入出力に基づき、まずテストを作成する
- 実装コードは書かず、テストのみを用意する
- テストを実行し、失敗を確認する
- テストが正しいことを確認できた段階でコミットする
- その後、テストをパスさせる実装を進める
- 実装中はテストを変更せず、コードを修正し続ける
- すべてのテストが通過するまで繰り返す

## 注意事項
- 機密情報の外部送信禁止
- 本番環境への直接操作禁止
- 未テストコードの本番適用禁止
