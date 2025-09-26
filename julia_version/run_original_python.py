#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
オリジナルのPythonコード（Numba最適化版）の実行方法説明と実行スクリプト
"""

import os
import sys
from pathlib import Path

print("=" * 80)
print("オリジナルPythonコード実行方法")
print("=" * 80)

# 実行に必要な条件と手順を表示
print("\n🔧 オリジナルのPython版を実行するには：")

print("\n【1. 必要なファイル】")
print("✅ org/IHCP_CGM_Sliding_Window_Calculation_ver2.py（メインスクリプト）")
print("✅ org/metal_thermal_properties.csv（SUS304熱物性データ）")
print("✅ T_measure_700um_1ms.npy（実測定データ 1.1GB）")
print("✅ MATLABファイル（IRカメラデータ SUS*.MAT形式）")

print("\n【2. 環境設定】")
print("・Python 3.12.7")
print("・NumPy 1.26.4")
print("・SciPy 1.13.1")
print("・Pandas 2.2.2")
print("・Numba 0.60.0（並列処理用）")

print("\n【3. 実行コマンド】")
print("cd ../org")
print("python IHCP_CGM_Sliding_Window_Calculation_ver2.py")

print("\n【4. オリジナルコードの特徴】")
print("・@njit(parallel=True) + prange による完全Numba最適化")
print("・スライディングウィンドウCGM実装")
print("・IRカメラ温度データの自動読み込み")
print("・表面熱流束の時系列逆解析")

print("\n【5. 実行時の注意点】")
print("・データファイルパスの確認（D:/HT_Calculation_Python/... を修正）")
print("・NUMBA_NUM_THREADS環境変数設定（推奨: 8）")
print("・十分なメモリ容量（推奨: 8GB以上）")
print("・実行時間: フルデータで数時間〜十数時間")

# パス確認
org_path = Path(__file__).parent.parent / 'org'
main_script = org_path / 'IHCP_CGM_Sliding_Window_Calculation_ver2.py'
thermal_data = org_path / 'metal_thermal_properties.csv'
measure_data = org_path / 'T_measure_700um_1ms.npy'

print("\n【6. ファイル存在確認】")
print(f"・メインスクリプト: {'✅' if main_script.exists() else '❌'} {main_script}")
print(f"・熱物性データ: {'✅' if thermal_data.exists() else '❌'} {thermal_data}")
print(f"・測定データ: {'✅' if measure_data.exists() else '❌'} {measure_data}")

print("\n【7. 最適化設定の比較】")
print("・オリジナル版: @njit(parallel=True) + prange（8スレッド）")
print("・Julia版: Threads.@threads（8スレッド）")
print("・性能: オリジナル版≒我々のNumba版≫Julia版")

print("\n【8. 実際の実行例】")
print("環境変数設定:")
print("export NUMBA_NUM_THREADS=8")
print("export OMP_NUM_THREADS=8")
print("")
print("実行:")
print("cd ../org")
print("python IHCP_CGM_Sliding_Window_Calculation_ver2.py")

# 簡単な実行チェック
if main_script.exists():
    print("\n【9. 簡易動作確認】")
    print("オリジナルスクリプトが見つかりました。")
    print("実行する場合は以下のコマンドを使用してください：")
    print(f"cd {org_path}")
    print("python IHCP_CGM_Sliding_Window_Calculation_ver2.py")

    # 依存関係確認
    try:
        import numba
        print(f"✅ Numba {numba.__version__} インストール済み")
        print(f"✅ Numbaスレッド数: {numba.get_num_threads()}")
    except ImportError:
        print("❌ Numba未インストール（pip install numba）")

    try:
        import scipy
        print(f"✅ SciPy {scipy.__version__} インストール済み")
    except ImportError:
        print("❌ SciPy未インストール（pip install scipy）")

else:
    print("\n❌ オリジナルスクリプトが見つかりません。")

print("\n" + "=" * 80)
print("Julia版との主な違い:")
print("・言語: Python（Numba JIT） vs Julia（ネイティブ）")
print("・並列化: prange vs @threads")
print("・Sparse行列: SciPy vs SparseArrays.jl")
print("・線形ソルバー: scipy.sparse.linalg.cg vs IterativeSolvers.cg")
print("・性能: ほぼ同等（適切な最適化により）")
print("=" * 80)