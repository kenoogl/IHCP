#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Python/Numba および Julia のスレッド数確認
"""

import os
import threading
import multiprocessing
import numpy as np
from numba import config, get_num_threads, set_num_threads

print("=" * 60)
print("Python/Numba スレッド数確認")
print("=" * 60)

# システム情報
print(f"CPU物理コア数: {multiprocessing.cpu_count()}")
print(f"システム論理プロセッサ数: {os.cpu_count()}")

# Python標準
print(f"Pythonデフォルトスレッド数: {threading.active_count()}")

# Numba設定
print(f"Numbaデフォルトスレッド数: {get_num_threads()}")
print(f"Numba設定 (config.NUMBA_DEFAULT_NUM_THREADS): {config.NUMBA_DEFAULT_NUM_THREADS}")
print(f"Numba設定 (config.NUMBA_NUM_THREADS): {config.NUMBA_NUM_THREADS}")

# 環境変数
env_vars = [
    'OMP_NUM_THREADS',
    'NUMBA_NUM_THREADS',
    'MKL_NUM_THREADS',
    'OPENBLAS_NUM_THREADS',
    'JULIA_NUM_THREADS'
]

print("\n環境変数:")
for var in env_vars:
    value = os.environ.get(var, "未設定")
    print(f"  {var}: {value}")

# NumPy/SciPy設定確認
try:
    import mkl
    print(f"\nIntel MKL最大スレッド数: {mkl.get_max_threads()}")
except ImportError:
    print("\nIntel MKL: 未インストール")

try:
    print(f"NumPy BLAS情報:")
    import numpy as np
    config_info = np.show_config()
    print("  BLAS設定確認完了")
except:
    print("  BLAS設定確認失敗")

# 実際のNumba並列実行テスト
print("\n実際のNumba並列実行テスト:")
from numba import njit, prange

@njit(parallel=True)
def parallel_test(n):
    result = 0.0
    for i in prange(n):
        result += i * i
    return result

# ウォームアップ
parallel_test(100)

# テスト実行
import time
n = 10000000
start_time = time.time()
result = parallel_test(n)
elapsed = time.time() - start_time

print(f"  並列計算テスト完了: {elapsed:.4f}秒")
print(f"  結果: {result}")