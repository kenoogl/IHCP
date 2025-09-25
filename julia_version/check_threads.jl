#!/usr/bin/env julia
# -*- coding: utf-8 -*-
"""
Juliaスレッド数確認
"""

println("=" ^ 60)
println("Julia スレッド数確認")
println("=" ^ 60)

# Julia基本情報
println("Julia基本スレッド情報:")
println("  Threads.nthreads(): ", Threads.nthreads())
println("  Threads.maxthreadid(): ", Threads.maxthreadid())

# システム情報
println("\nシステム情報:")
println("  Sys.CPU_THREADS: ", Sys.CPU_THREADS)

# 環境変数
println("\nJulia関連環境変数:")
env_vars = ["JULIA_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"]
for var in env_vars
    value = get(ENV, var, "未設定")
    println("  $var: $value")
end

# BLAS情報
using LinearAlgebra
println("\nBLAS設定:")
println("  BLAS.get_num_threads(): ", BLAS.get_num_threads())
println("  BLAS ライブラリ: ", BLAS.get_config())

# 実際の並列実行テスト
println("\n実際のJulia並列実行テスト:")

function parallel_test_julia(n)
    result = 0.0
    Threads.@threads for i in 1:n
        result += i * i  # レースコンディション注意（テスト目的）
    end
    return result
end

# より安全な並列テスト
function parallel_test_safe(n)
    results = Vector{Float64}(undef, Threads.nthreads())
    Threads.@threads for tid in 1:Threads.nthreads()
        local_sum = 0.0
        start_idx = div((tid-1) * n, Threads.nthreads()) + 1
        end_idx = div(tid * n, Threads.nthreads())
        for i in start_idx:end_idx
            local_sum += i * i
        end
        results[tid] = local_sum
    end
    return sum(results)
end

# ウォームアップ
parallel_test_safe(100)

# テスト実行
n = 10000000
start_time = time()
result = parallel_test_safe(n)
elapsed = time() - start_time

println("  並列計算テスト完了: $(elapsed)秒")
println("  結果: $result")

# スレッドID確認
println("\nスレッド動作確認:")
thread_ids = Vector{Int}(undef, Threads.nthreads())
Threads.@threads for i in 1:Threads.nthreads()
    thread_ids[i] = Threads.threadid()
end
println("  アクティブスレッドID: ", sort(thread_ids))