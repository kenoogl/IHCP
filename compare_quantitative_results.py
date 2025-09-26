#!/usr/bin/env python3

"""
Python版とJulia版の定量的比較レポート生成
フルサイズ10ステップ計算結果の詳細比較
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

print("=" * 80)
print("Python版とJulia版の定量的一致性比較レポート")
print("フルサイズ10ステップ計算結果の詳細分析")
print("=" * 80)

def load_and_compare_results():
    """結果ファイルを読み込んで比較分析"""

    # Julia版結果読み込み
    try:
        julia_dhcp = np.load("julia_version/julia_dhcp_results_fullsize.npz")
        julia_adjoint = np.load("julia_version/julia_adjoint_results_fullsize.npz")
        julia_sensitivity = np.load("julia_version/julia_sensitivity_results_fullsize.npz")
        print("✓ Julia版結果ファイル読み込み成功")
    except Exception as e:
        print(f"✗ Julia版結果読み込みエラー: {e}")
        return None

    # Python版結果読み込み
    try:
        python_dhcp = np.load("org/python_dhcp_results_fullsize.npz")
        python_adjoint = np.load("org/python_adjoint_results_fullsize.npz")
        python_sensitivity = np.load("org/python_sensitivity_results_fullsize.npz")
        print("✓ Python版結果ファイル読み込み成功")
    except Exception as e:
        print(f"✗ Python版結果読み込みエラー: {e}")
        return None

    return {
        'julia_dhcp': julia_dhcp,
        'julia_adjoint': julia_adjoint,
        'julia_sensitivity': julia_sensitivity,
        'python_dhcp': python_dhcp,
        'python_adjoint': python_adjoint,
        'python_sensitivity': python_sensitivity
    }

def analyze_dhcp_results(results):
    """順問題結果の定量的比較"""
    print("\n【順問題（DHCP）結果比較】")
    print("=" * 50)

    T_julia = results['julia_dhcp']['T_result']
    T_python = results['python_dhcp']['T_result']

    print(f"Julia版配列形状: {T_julia.shape}")
    print(f"Python版配列形状: {T_python.shape}")

    # 絶対誤差・相対誤差計算
    abs_error = np.abs(T_julia - T_python)
    rel_error = abs_error / (np.abs(T_python) + 1e-10)

    print(f"\n温度結果統計:")
    print(f"  Julia版温度範囲: {np.min(T_julia):.4f} - {np.max(T_julia):.4f} K")
    print(f"  Python版温度範囲: {np.min(T_python):.4f} - {np.max(T_python):.4f} K")

    print(f"\n誤差統計:")
    print(f"  最大絶対誤差: {np.max(abs_error):.2e} K")
    print(f"  平均絶対誤差: {np.mean(abs_error):.2e} K")
    print(f"  最大相対誤差: {np.max(rel_error):.2e}")
    print(f"  平均相対誤差: {np.mean(rel_error):.2e}")

    print(f"\n計算時間比較:")
    julia_time = float(results['julia_dhcp']['computation_time'])
    python_time = float(results['python_dhcp']['computation_time'])
    print(f"  Julia版計算時間: {julia_time:.3f}秒")
    print(f"  Python版計算時間: {python_time:.3f}秒")
    print(f"  性能比（Python/Julia): {python_time/julia_time:.2f}x")

    # 一致性判定
    if np.max(rel_error) < 1e-10:
        print(f"  ✓ 順問題結果: 高精度一致（相対誤差 < 1e-10）")
    elif np.max(rel_error) < 1e-6:
        print(f"  ✓ 順問題結果: 良好な一致（相対誤差 < 1e-6）")
    else:
        print(f"  ⚠ 順問題結果: 要確認（相対誤差 = {np.max(rel_error):.2e}）")

    return {
        'max_abs_error': np.max(abs_error),
        'max_rel_error': np.max(rel_error),
        'mean_abs_error': np.mean(abs_error),
        'mean_rel_error': np.mean(rel_error),
        'julia_time': julia_time,
        'python_time': python_time
    }

def analyze_adjoint_results(results):
    """随伴問題結果の定量的比較"""
    print("\n【随伴問題結果比較】")
    print("=" * 50)

    adj_julia = results['julia_adjoint']['adjoint_result']
    adj_python = results['python_adjoint']['adjoint_result']

    print(f"Julia版配列形状: {adj_julia.shape}")
    print(f"Python版配列形状: {adj_python.shape}")

    # 絶対誤差・相対誤差計算
    abs_error = np.abs(adj_julia - adj_python)
    rel_error = abs_error / (np.abs(adj_python) + 1e-15)

    print(f"\n随伴変数統計:")
    print(f"  Julia版範囲: {np.min(adj_julia):.2e} - {np.max(adj_julia):.2e}")
    print(f"  Python版範囲: {np.min(adj_python):.2e} - {np.max(adj_python):.2e}")

    print(f"\n誤差統計:")
    print(f"  最大絶対誤差: {np.max(abs_error):.2e}")
    print(f"  平均絶対誤差: {np.mean(abs_error):.2e}")
    print(f"  最大相対誤差: {np.max(rel_error):.2e}")
    print(f"  平均相対誤差: {np.mean(rel_error):.2e}")

    print(f"\n計算時間比較:")
    julia_time = float(results['julia_adjoint']['computation_time'])
    python_time = float(results['python_adjoint']['computation_time'])
    print(f"  Julia版計算時間: {julia_time:.3f}秒")
    print(f"  Python版計算時間: {python_time:.3f}秒")
    print(f"  性能比（Python/Julia): {python_time/julia_time:.2f}x")

    # 一致性判定
    if np.max(rel_error) < 1e-10:
        print(f"  ✓ 随伴問題結果: 高精度一致（相対誤差 < 1e-10）")
    elif np.max(rel_error) < 1e-6:
        print(f"  ✓ 随伴問題結果: 良好な一致（相対誤差 < 1e-6）")
    else:
        print(f"  ⚠ 随伴問題結果: 要確認（相対誤差 = {np.max(rel_error):.2e}）")

    return {
        'max_abs_error': np.max(abs_error),
        'max_rel_error': np.max(rel_error),
        'mean_abs_error': np.mean(abs_error),
        'mean_rel_error': np.mean(rel_error),
        'julia_time': julia_time,
        'python_time': python_time
    }

def analyze_sensitivity_results(results):
    """感度問題結果の定量的比較"""
    print("\n【感度問題結果比較】")
    print("=" * 50)

    sens_julia = results['julia_sensitivity']['sensitivity']
    sens_python = results['python_sensitivity']['sensitivity']

    print(f"Julia版配列形状: {sens_julia.shape}")
    print(f"Python版配列形状: {sens_python.shape}")

    # 絶対誤差・相対誤差計算
    abs_error = np.abs(sens_julia - sens_python)
    rel_error = abs_error / (np.abs(sens_python) + 1e-15)

    print(f"\n感度統計:")
    print(f"  Julia版範囲: {np.min(sens_julia):.2e} - {np.max(sens_julia):.2e}")
    print(f"  Python版範囲: {np.min(sens_python):.2e} - {np.max(sens_python):.2e}")

    print(f"\n誤差統計:")
    print(f"  最大絶対誤差: {np.max(abs_error):.2e}")
    print(f"  平均絶対誤差: {np.mean(abs_error):.2e}")
    print(f"  最大相対誤差: {np.max(rel_error):.2e}")
    print(f"  平均相対誤差: {np.mean(rel_error):.2e}")

    print(f"\n計算時間比較:")
    julia_time = float(results['julia_sensitivity']['computation_time'])
    python_time = float(results['python_sensitivity']['computation_time'])
    print(f"  Julia版計算時間: {julia_time:.3f}秒")
    print(f"  Python版計算時間: {python_time:.3f}秒")
    print(f"  性能比（Python/Julia): {python_time/julia_time:.2f}x")

    # 一致性判定
    if np.max(rel_error) < 1e-10:
        print(f"  ✓ 感度問題結果: 高精度一致（相対誤差 < 1e-10）")
    elif np.max(rel_error) < 1e-6:
        print(f"  ✓ 感度問題結果: 良好な一致（相対誤差 < 1e-6）")
    else:
        print(f"  ⚠ 感度問題結果: 要確認（相対誤差 = {np.max(rel_error):.2e}）")

    return {
        'max_abs_error': np.max(abs_error),
        'max_rel_error': np.max(rel_error),
        'mean_abs_error': np.mean(abs_error),
        'mean_rel_error': np.mean(rel_error),
        'julia_time': julia_time,
        'python_time': python_time
    }

def generate_summary_report(dhcp_stats, adjoint_stats, sens_stats):
    """総合レポート生成"""
    print("\n" + "=" * 80)
    print("【総合比較レポート】")
    print("=" * 80)

    # 精度総括
    print("\n◆ 数値精度総括:")
    problems = ['順問題', '随伴問題', '感度問題']
    stats_list = [dhcp_stats, adjoint_stats, sens_stats]

    print(f"{'問題種類':<10} {'最大相対誤差':<15} {'平均相対誤差':<15} {'判定':<10}")
    print("-" * 60)

    all_accurate = True
    for i, (problem, stats) in enumerate(zip(problems, stats_list)):
        max_rel = stats['max_rel_error']
        mean_rel = stats['mean_rel_error']

        if max_rel < 1e-10:
            judgment = "高精度一致"
        elif max_rel < 1e-6:
            judgment = "良好"
        else:
            judgment = "要確認"
            all_accurate = False

        print(f"{problem:<10} {max_rel:<15.2e} {mean_rel:<15.2e} {judgment:<10}")

    print(f"\n◆ 性能比較総括:")
    print(f"{'問題種類':<10} {'Julia時間[秒]':<12} {'Python時間[秒]':<13} {'性能比':<8}")
    print("-" * 55)

    total_julia_time = 0
    total_python_time = 0

    for problem, stats in zip(problems, stats_list):
        julia_t = stats['julia_time']
        python_t = stats['python_time']
        ratio = python_t / julia_t

        total_julia_time += julia_t
        total_python_time += python_t

        print(f"{problem:<10} {julia_t:<12.3f} {python_t:<13.3f} {ratio:<8.2f}x")

    total_ratio = total_python_time / total_julia_time
    print("-" * 55)
    print(f"{'合計':<10} {total_julia_time:<12.3f} {total_python_time:<13.3f} {total_ratio:<8.2f}x")

    print(f"\n◆ 最終判定:")
    if all_accurate:
        print("✅ Python版とJulia版は定量的に高精度で一致しています")
        print("   すべての問題で相対誤差 < 1e-6 を達成")
    else:
        print("⚠️  一部の問題で精度に課題があります")
        print("   詳細な調査が必要です")

    print(f"\n◆ 性能評価:")
    if total_ratio > 2.0:
        print(f"⚡ Julia版はPython版より{total_ratio:.1f}倍高速です")
    elif total_ratio > 1.5:
        print(f"📈 Julia版はPython版より{total_ratio:.1f}倍高速です")
    else:
        print(f"📊 両版の性能は同等です（比率: {total_ratio:.1f}x）")

def main():
    """メイン実行"""
    print(f"実行日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 結果読み込み
    results = load_and_compare_results()
    if results is None:
        print("エラー: 結果ファイルが見つかりません")
        print("先にJulia版とPython版の計算を実行してください")
        return

    # 各問題の比較分析
    dhcp_stats = analyze_dhcp_results(results)
    adjoint_stats = analyze_adjoint_results(results)
    sens_stats = analyze_sensitivity_results(results)

    # 総合レポート
    generate_summary_report(dhcp_stats, adjoint_stats, sens_stats)

    print(f"\n比較レポート完了: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("詳細な数値データは各結果ファイル(.npz)で確認できます")

if __name__ == "__main__":
    main()