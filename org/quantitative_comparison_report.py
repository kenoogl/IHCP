#!/usr/bin/env python3

"""
Python版とJulia版の定量的比較レポート
テストケース1の詳細分析
"""

import numpy as np
from datetime import datetime

print("="*80)
print("Python版とJulia版の定量的一致性検証レポート")
print("テストケース1結果の詳細比較分析")
print("="*80)

def load_results():
    """結果ファイルを読み込み"""
    try:
        julia_results = np.load("../julia_version/test_case1_julia_results.npz")
        python_results = np.load("test_case1_python_results.npz")
        print("✓ 結果ファイル読み込み成功")
        return julia_results, python_results
    except Exception as e:
        print(f"✗ 結果ファイル読み込みエラー: {e}")
        return None, None

def analyze_computation_times(julia_results, python_results):
    """計算時間の比較"""
    print("\n【計算時間比較】")
    print("="*50)

    julia_time = float(julia_results['elapsed_time'])
    python_time = float(python_results['elapsed_time'])

    print(f"Julia版実行時間:  {julia_time:.6f}秒")
    print(f"Python版実行時間: {python_time:.6f}秒")
    print(f"性能比（Python/Julia): {python_time/julia_time:.2f}x")

    if python_time < julia_time:
        print(f"✅ Python版が{julia_time/python_time:.1f}倍高速")
    else:
        print(f"⚡ Julia版が{python_time/julia_time:.1f}倍高速")

    return {'julia_time': julia_time, 'python_time': python_time, 'ratio': python_time/julia_time}

def analyze_convergence(julia_results, python_results):
    """収束解析の比較"""
    print("\n【収束解析比較】")
    print("="*50)

    # 目的関数の比較
    julia_obj_initial = float(julia_results['objective_initial'])
    julia_obj_final = float(julia_results['objective_final'])
    python_obj_initial = float(python_results['objective_initial'])
    python_obj_final = float(python_results['objective_final'])

    print(f"Julia版:")
    print(f"  初期目的関数: {julia_obj_initial:.6e}")
    print(f"  最終目的関数: {julia_obj_final:.6e}")
    print(f"  相対改善率: {(julia_obj_initial-julia_obj_final)/julia_obj_initial:.6f}")

    print(f"Python版:")
    print(f"  初期目的関数: {python_obj_initial:.6e}")
    print(f"  最終目的関数: {python_obj_final:.6e}")
    print(f"  相対改善率: {(python_obj_initial-python_obj_final)/python_obj_initial:.6f}")

    # 最終目的関数の比較
    final_diff = abs(julia_obj_final - python_obj_final)
    relative_diff = final_diff / min(julia_obj_final, python_obj_final)

    print(f"\n目的関数の差異:")
    print(f"  絶対誤差: {final_diff:.6e}")
    print(f"  相対誤差: {relative_diff:.6e}")

    if relative_diff < 1e-6:
        print("✅ 目的関数は高精度で一致")
    elif relative_diff < 1e-3:
        print("✓ 目的関数は良好な一致")
    else:
        print("⚠ 目的関数に有意な差異")

    return {
        'julia_final': julia_obj_final,
        'python_final': python_obj_final,
        'relative_diff': relative_diff
    }

def analyze_heat_flux(julia_results, python_results):
    """熱流束結果の比較"""
    print("\n【熱流束比較】")
    print("="*50)

    # 統計データから熱流束を比較（実際の配列データがない場合）
    julia_q_min = float(julia_results['q_min'])
    julia_q_max = float(julia_results['q_max'])
    julia_q_mean = float(julia_results['q_mean'])
    julia_q_std = float(julia_results['q_std'])

    python_q_min = float(python_results['q_min'])
    python_q_max = float(python_results['q_max'])
    python_q_mean = float(python_results['q_mean'])
    python_q_std = float(python_results['q_std'])

    print(f"格子サイズ:")
    print(f"  Julia版: {julia_results['grid_size']}")
    print(f"  Python版: {python_results['grid_size']}")

    # 統計値比較
    print(f"\nJulia版統計:")
    print(f"  最小値: {julia_q_min:.6e} W/m²")
    print(f"  最大値: {julia_q_max:.6e} W/m²")
    print(f"  平均値: {julia_q_mean:.6e} W/m²")
    print(f"  標準偏差: {julia_q_std:.6e} W/m²")

    print(f"\nPython版統計:")
    print(f"  最小値: {python_q_min:.6e} W/m²")
    print(f"  最大値: {python_q_max:.6e} W/m²")
    print(f"  平均値: {python_q_mean:.6e} W/m²")
    print(f"  標準偏差: {python_q_std:.6e} W/m²")

    # 統計値から誤差を計算
    mean_abs_error = abs(julia_q_mean - python_q_mean)
    mean_rel_error = mean_abs_error / (abs(python_q_mean) + 1e-15)

    std_abs_error = abs(julia_q_std - python_q_std)
    std_rel_error = std_abs_error / (abs(python_q_std) + 1e-15)

    range_julia = julia_q_max - julia_q_min
    range_python = python_q_max - python_q_min
    range_abs_error = abs(range_julia - range_python)
    range_rel_error = range_abs_error / (abs(range_python) + 1e-15)

    print(f"\n統計値誤差:")
    print(f"  平均値絶対誤差: {mean_abs_error:.6e} W/m²")
    print(f"  平均値相対誤差: {mean_rel_error:.6e}")
    print(f"  標準偏差絶対誤差: {std_abs_error:.6e} W/m²")
    print(f"  標準偏差相対誤差: {std_rel_error:.6e}")
    print(f"  レンジ絶対誤差: {range_abs_error:.6e} W/m²")
    print(f"  レンジ相対誤差: {range_rel_error:.6e}")

    # 一致性判定（統計値ベース）
    max_rel_error = max(mean_rel_error, std_rel_error, range_rel_error)

    if max_rel_error < 1e-10:
        print("✅ 熱流束統計は高精度で一致（相対誤差 < 1e-10）")
        consistency = "高精度一致"
    elif max_rel_error < 1e-6:
        print("✓ 熱流束統計は良好な一致（相対誤差 < 1e-6）")
        consistency = "良好一致"
    elif max_rel_error < 1e-3:
        print("○ 熱流束統計は妥当な一致（相対誤差 < 1e-3）")
        consistency = "妥当一致"
    else:
        print("⚠ 熱流束統計に有意な差異")
        consistency = "有意差異"

    return {
        'max_abs_error': max(mean_abs_error, std_abs_error, range_abs_error),
        'max_rel_error': max_rel_error,
        'mean_abs_error': mean_abs_error,
        'mean_rel_error': mean_rel_error,
        'consistency': consistency
    }

def analyze_temperature_prediction(julia_results, python_results):
    """温度予測精度の比較"""
    print("\n【温度予測精度比較】")
    print("="*50)

    # 予測精度メトリクス
    julia_rmse = float(julia_results['temp_rmse'])
    julia_mae = float(julia_results['temp_mae'])
    julia_max_error = float(julia_results['temp_max_error'])

    python_rmse = float(python_results['temp_rmse'])
    python_mae = float(python_results['temp_mae'])
    python_max_error = float(python_results['temp_max_error'])

    print(f"Julia版:")
    print(f"  RMSE: {julia_rmse:.6f} K")
    print(f"  MAE: {julia_mae:.6f} K")
    print(f"  最大誤差: {julia_max_error:.6f} K")

    print(f"Python版:")
    print(f"  RMSE: {python_rmse:.6f} K")
    print(f"  MAE: {python_mae:.6f} K")
    print(f"  最大誤差: {python_max_error:.6f} K")

    print(f"\n精度差異:")
    print(f"  RMSE差: {abs(julia_rmse - python_rmse):.6f} K")
    print(f"  MAE差: {abs(julia_mae - python_mae):.6f} K")
    print(f"  最大誤差差: {abs(julia_max_error - python_max_error):.6f} K")

    return {
        'julia_rmse': julia_rmse,
        'python_rmse': python_rmse,
        'julia_mae': julia_mae,
        'python_mae': python_mae
    }

def generate_final_assessment(time_stats, conv_stats, flux_stats, temp_stats):
    """最終評価レポート生成"""
    print("\n" + "="*80)
    print("【総合評価レポート】")
    print("="*80)

    print("\n◆ 数値精度評価:")
    assessments = []

    # 収束評価
    if conv_stats['relative_diff'] < 1e-10:
        conv_grade = "A+ (高精度一致)"
        assessments.append(('収束精度', 'A+'))
    elif conv_stats['relative_diff'] < 1e-6:
        conv_grade = "A (良好一致)"
        assessments.append(('収束精度', 'A'))
    else:
        conv_grade = "B (要改善)"
        assessments.append(('収束精度', 'B'))

    print(f"  収束精度: {conv_grade}")

    # 熱流束評価
    if 'consistency' in flux_stats:
        flux_grade = {
            '高精度一致': 'A+',
            '良好一致': 'A',
            '妥当一致': 'B',
            '有意差異': 'C'
        }.get(flux_stats['consistency'], 'C')
        print(f"  熱流束精度: {flux_grade} ({flux_stats['consistency']})")
        assessments.append(('熱流束精度', flux_grade))

    print(f"\n◆ 性能評価:")
    if time_stats['ratio'] < 0.5:
        perf_grade = "A+ (Python高速)"
    elif time_stats['ratio'] < 1.0:
        perf_grade = "A (Python優勢)"
    elif time_stats['ratio'] < 2.0:
        perf_grade = "B (Julia優勢)"
    else:
        perf_grade = "A (Julia高速)"

    print(f"  実行性能: {perf_grade} (比率: {time_stats['ratio']:.2f}x)")
    assessments.append(('実行性能', perf_grade.split()[0]))

    print(f"\n◆ 物理的妥当性評価:")
    # 熱流束の物理的妥当性
    if 'max_abs_error' in flux_stats:
        if flux_stats['max_abs_error'] < 1e3:  # 1 kW/m²以下
            physics_grade = "A (物理的妥当)"
        elif flux_stats['max_abs_error'] < 1e5:  # 100 kW/m²以下
            physics_grade = "B (許容範囲)"
        else:
            physics_grade = "C (要確認)"
        print(f"  熱流束妥当性: {physics_grade}")
        assessments.append(('物理的妥当性', physics_grade.split()[0]))

    print(f"\n◆ 総合判定:")
    grades = [grade for _, grade in assessments]
    avg_score = sum({'A+': 5, 'A': 4, 'B': 3, 'C': 2, 'D': 1}.get(g, 1) for g in grades) / len(grades)

    if avg_score >= 4.5:
        overall_grade = "A+ - 優秀"
        verdict = "✅ Python版とJulia版は数値的に高精度で一致しており、両実装は信頼性が高い"
    elif avg_score >= 3.5:
        overall_grade = "A - 良好"
        verdict = "✓ Python版とJulia版は良好に一致しており、実用上問題なし"
    elif avg_score >= 2.5:
        overall_grade = "B - 妥当"
        verdict = "○ Python版とJulia版は妥当に一致しているが、一部改善の余地あり"
    else:
        overall_grade = "C - 要改善"
        verdict = "⚠ Python版とJulia版の一致性に課題があり、詳細調査が必要"

    print(f"  総合評価: {overall_grade}")
    print(f"  {verdict}")

    return overall_grade, verdict

def main():
    """メイン実行"""
    print(f"レポート作成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 結果読み込み
    julia_results, python_results = load_results()
    if julia_results is None or python_results is None:
        print("エラー: 結果ファイルが見つかりません")
        return

    # 各項目の比較分析
    time_stats = analyze_computation_times(julia_results, python_results)
    conv_stats = analyze_convergence(julia_results, python_results)
    flux_stats = analyze_heat_flux(julia_results, python_results)
    temp_stats = analyze_temperature_prediction(julia_results, python_results)

    # 総合評価
    overall_grade, verdict = generate_final_assessment(time_stats, conv_stats, flux_stats, temp_stats)

    print(f"\n定量的比較レポート完了: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"結果: {overall_grade}")

if __name__ == "__main__":
    main()