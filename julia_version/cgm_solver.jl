# -*- coding: utf-8 -*-
"""
共役勾配法（CGM: Conjugate Gradient Method）ソルバー
逆熱伝導問題の最適化求解
Python版からの変換: 表面熱流束の最適化計算
"""

include("adjoint_solver.jl")

using Printf

"""
内積計算（テンソル用）
Python の _dot 相当
"""
function tensor_dot(a::Array{Float64}, b::Array{Float64})
    return sum(a .* b)
end

"""
単一計算ウィンドウでのグローバルCGM計算
Python の global_CGM_time 相当

引数:
- T_init: 初期温度分布 (ni, nj, nk)
- Y_obs: 観測温度分布 (nt, ni, nj)
- q_init: 初期熱流束推定値 (nt-1, ni, nj)
- dx, dy: 格子間隔
- dz, dz_b, dz_t: z方向格子パラメータ
- dt: 時間ステップ
- rho: 密度
- cp_coeffs, k_coeffs: 熱物性値係数
- CGM_iteration: 最大CGM反復数 (デフォルト=20000)

戻り値:
- q: 最適化された表面熱流束 (nt-1, ni, nj)
- T_final: 最終温度分布 (ni, nj, nk)
- J_hist: 目的関数履歴
"""
function global_CGM_time(T_init::Array{Float64,3}, Y_obs::Array{Float64,3}, q_init::Array{Float64,3},
                         dx::Float64, dy::Float64, dz::Vector{Float64},
                         dz_b::Vector{Float64}, dz_t::Vector{Float64}, dt::Float64,
                         rho::Float64, cp_coeffs::Vector{Float64}, k_coeffs::Vector{Float64};
                         CGM_iteration::Int=20000)

    nt = size(Y_obs, 1)
    ni, nj, nk = size(T_init)
    q = copy(q_init)

    J_hist = Float64[]  # 目的関数履歴

    # CGMパラメータ
    M = ni * nj
    sigma = 1.8  # 測定誤差仮定
    epsilon = M * (sigma^2) * (nt - 1)  # 反復停止基準

    grad = zeros(size(q))          # 勾配
    grad_last = zeros(size(q))     # 前回の勾配
    p_n_last = zeros(size(q))      # 前回の探索方向

    # 方向リセット間隔
    dire_reset_every = 5

    # 平台検出パラメータ
    P = 10        # 近P次平均
    eta = 1e-4    # 平均相対下降閾値
    min_iter = 10 # 最小反復数

    eps = 1e-15  # 数値安定性のための最小値（Julia版では必要）

    println("CGM最適化開始")
    println(@sprintf("格子サイズ: %d×%d×%d, 時間ステップ: %d", ni, nj, nk, nt))
    println(@sprintf("停止基準: ε=%.2e, 最小反復: %d", epsilon, min_iter))

    T_cal = nothing  # 初期化

    for it in 1:CGM_iteration
        t0 = time()

        # Step 1: 直接問題解法（全時間ステップ）
        T_cal = multiple_time_step_solver_DHCP(
            T_init, q, nt, rho, cp_coeffs, k_coeffs,
            dx, dy, dz, dz_b, dz_t, dt,
            1e-6, 20000
        )

        # Step 2: 停止基準チェック
        res_T = T_cal[2:end, :, :, 1] - Y_obs[2:end, :, :]  # 温度残差（表面）
        delta_T = abs.(res_T)
        J = tensor_dot(res_T, res_T)  # 目的関数
        push!(J_hist, J)

        # 不一致原理（Discrepancy Principle）
        if it >= min_iter && J < epsilon && maximum(delta_T) <= sigma
            println(@sprintf("[停止] 不一致原理満足: J=%.4e < %.4e, max|ΔT|=%.3e ≤ σ=%.1f",
                           J, epsilon, maximum(delta_T), sigma))
            break
        end

        # 平台検出
        rel_drop_avg = nothing
        if length(J_hist) >= P + 1
            drops = Float64[]
            for i in (length(J_hist)-P+1):length(J_hist)
                prev_val, cur_val = J_hist[i-1], J_hist[i]
                push!(drops, max(0.0, (prev_val - cur_val) / (abs(prev_val) + eps)))
            end
            rel_drop_avg = sum(drops) / P
        end

        if it >= min_iter && rel_drop_avg !== nothing && rel_drop_avg < eta
            println(@sprintf("[停止] 平台検出: rel_drop_avg=%.3e < η=%.1e (近%d步平均進展過小)",
                           rel_drop_avg, eta, P))
            break
        end

        # Step 3: 隨伴問題解法
        lambda_field = multiple_time_step_solver_Adjoint(
            T_cal, Y_obs, nt, rho, cp_coeffs, k_coeffs,
            dx, dy, dz, dz_b, dz_t, dt,
            1e-8, 20000
        )

        if !all(isfinite.(lambda_field))
            println(@sprintf("[警告] 隨伴場にNaN/Inf at iter %d", it))
        end

        # Step 4: 勾配計算（表面隨伴変数）
        for n in 1:(nt-1)
            grad[n, :, :] = lambda_field[n, :, :, nk]  # 上面（z最大）
        end

        # Step 5: 共役勾配方向計算
        if it == 1 || tensor_dot(grad, p_n_last) <= 0 || it % dire_reset_every == 0
            p_n = copy(grad)
            gamma = 0.0
        else
            y = grad - grad_last
            denom = tensor_dot(grad_last, grad_last) + eps
            gamma = max(0.0, tensor_dot(grad, y) / denom)
            p_n_candidate = grad + gamma * p_n_last

            if tensor_dot(grad, p_n_candidate) > 0
                p_n = p_n_candidate
            else
                p_n = copy(grad)
            end
        end

        p_n_last = copy(p_n)

        # Step 6: 感度問題解法（δq = p_n）
        dT_init = zeros(size(T_init))
        dT = multiple_time_step_solver_DHCP(
            dT_init, p_n, nt, rho, cp_coeffs, k_coeffs,
            dx, dy, dz, dz_b, dz_t, dt,
            1e-8, 20000
        )

        # Step 7: 最適ステップサイズ探索
        Sp = dT[1:(end-1), :, :, 1]  # 感度表面温度（Python版のdT[1:, :, :, bottom_idx]に対応）
        numerator = tensor_dot(res_T, Sp)
        denominator = tensor_dot(Sp, Sp)

        # 分母が極小の場合の処理（Julia版では必要）
        min_denominator = 1e-20
        if denominator < min_denominator
            println(@sprintf("  [警告] 分母が極小 %.2e < %.2e at iter %d", denominator, min_denominator, it))
            # 小さな勾配方向ステップを使用
            beta = 1e-6
        else
            beta = numerator / (denominator + eps)
        end

        # ステップサイズ制限（Python版より保守的にJulia版では調整）
        beta_max = 1e6  # Julia版では数値的により安定
        if it == 1 && abs(beta) > beta_max
            println(@sprintf("  [警告] βクリップ: %.2e => %.2e", beta, sign(beta) * beta_max))
            beta = clamp(beta, -beta_max, beta_max)
        end

        # 相対減少率計算
        rel_drop = nothing
        if length(J_hist) >= 2
            rel_drop = abs(J_hist[end] - J_hist[end-1]) / J_hist[end-1]
        end

        wall_s = time() - t0

        # 進捗出力
        println(@sprintf("@ ___ Iter %3d ___ @ wall_s = %.3fs", it, wall_s))
        println(@sprintf("J = %.5e, β = %.4e, rel_drop = %s",
                        J, beta, rel_drop === nothing ? "N/A" : @sprintf("%.3e", rel_drop)))
        println(@sprintf("|T - Y|: max=%.3e, min=%.3e, mean=%.3e",
                        maximum(delta_T), minimum(delta_T), sum(delta_T)/length(delta_T)))
        println(@sprintf("grad: min=%.4e, max=%.4e, mean=%.4e",
                        minimum(grad), maximum(grad), sum(grad)/length(grad)))
        println(@sprintf("dT:   min=%.4e, max=%.4e, mean=%.4e",
                        minimum(dT[2:end]), maximum(dT[2:end]), sum(dT[2:end])/length(dT[2:end])))
        println(@sprintf("q:    min=%.4e, max=%.4e, mean=%.4e",
                        minimum(q), maximum(q), sum(q)/length(q)))
        println(@sprintf("denominator at iter %d: %.4e", it, denominator))

        # q更新
        q = q - beta * p_n
        grad_last = copy(grad)
    end

    # 最終温度分布を安全に取得
    final_temp = T_cal !== nothing ? T_cal[end, :, :, :] : T_init
    return q, final_temp, J_hist
end

"""
スライディングウィンドウCGM計算（全時間領域）
Python の sliding_window_CGM_q_saving 相当

引数:
- Y_obs: 観測温度分布 (nt, ni, nj)
- T0: 初期温度分布 (ni, nj, nk)
- dx, dy: 格子間隔
- dz, dz_b, dz_t: z方向格子パラメータ
- dt: 時間ステップ
- rho: 密度
- cp_coeffs, k_coeffs: 熱物性値係数
- window_size: ウィンドウサイズ
- overlap: 重複ステップ数
- q_init_value: 初期熱流束値
- filename: 保存ファイル名
- CGM_iteration: 最大CGM反復数

戻り値:
- q_global: 全時間領域の最適熱流束
"""
function sliding_window_CGM_q_saving(Y_obs::Array{Float64,3}, T0::Array{Float64,3},
                                     dx::Float64, dy::Float64, dz::Vector{Float64},
                                     dz_b::Vector{Float64}, dz_t::Vector{Float64}, dt::Float64,
                                     rho::Float64, cp_coeffs::Vector{Float64}, k_coeffs::Vector{Float64},
                                     window_size::Int, overlap::Int, q_init_value::Float64, filename::String;
                                     CGM_iteration::Int=20000)

    nt = size(Y_obs, 1)
    T_init = copy(T0)
    ni, nj, nk = size(T_init)

    start_idx = 1  # Julia 1-based
    q_total = Array{Float64,3}[]
    prev_q_win = nothing

    safety_counter = 0
    safety_limit = nt * 5

    println("スライディングウィンドウCGM開始")
    println(@sprintf("全時間ステップ: %d, ウィンドウサイズ: %d, 重複: %d", nt, window_size, overlap))

    while start_idx < nt
        safety_counter += 1
        if safety_counter > safety_limit
            println("セーフティブレーク: 反復回数過多、重複/ウィンドウ設定を確認")
            break
        end

        # 現在の利用可能ウィンドウ長
        max_L = min(window_size, nt - start_idx)
        end_idx = start_idx + max_L
        Y_obs_win = Y_obs[start_idx:end_idx, :, :]

        # 当該ウィンドウの初期熱流束
        if prev_q_win === nothing
            q_init_win = fill(q_init_value, max_L, ni, nj)
        else
            q_init_win = Array{Float64}(undef, max_L, ni, nj)
            L_overlap = min(overlap, max_L, size(prev_q_win, 1))
            if L_overlap > 0
                q_init_win[1:L_overlap, :, :] = prev_q_win[end-L_overlap+1:end, :, :]
            end
            if L_overlap < max_L
                edge = prev_q_win[end, :, :]
                for i in (L_overlap+1):max_L
                    q_init_win[i, :, :] = edge
                end
            end
        end

        start_time_one_window = time()
        q_win, T_win_last, J_hist = global_CGM_time(
            T_init, Y_obs_win, q_init_win, dx, dy, dz, dz_b, dz_t, dt,
            rho, cp_coeffs, k_coeffs; CGM_iteration=CGM_iteration
        )
        end_time_one_window = time()

        prev_q_win = copy(q_win)

        # q拼接（重複部分平均化）
        if length(q_total) == 0
            push!(q_total, q_win)
        else
            overlap_steps = min(overlap, size(q_win, 1), size(q_total[end], 1))
            if overlap_steps > 0
                # 重複部分平均化
                avg_part = 0.5 * q_total[end][end-overlap_steps+1:end, :, :] +
                          0.5 * q_win[1:overlap_steps, :, :]
                q_total[end][end-overlap_steps+1:end, :, :] = avg_part

                if size(q_win, 1) > overlap_steps
                    push!(q_total, q_win[overlap_steps+1:end, :, :])
                end
            else
                push!(q_total, q_win)
            end
        end

        # 温度初期値更新
        if ndims(T_win_last) == 3
            T_init = copy(T_win_last)
        else
            T_init = copy(T_win_last[end, :, :, :])
        end

        println(@sprintf("ウィンドウ %.3f - %.3f 完了。J = %.3f, 時間 = %.2fs",
                        (start_idx-1)*dt, (end_idx-1)*dt, J_hist[end],
                        end_time_one_window - start_time_one_window))

        step = max(1, max_L - overlap)
        start_idx += step
    end

    # 全グローバルq拼接とクリップ
    q_global = cat(q_total...; dims=1)[1:(nt-1), :, :]

    # NPZファイルに保存（.jl拡張子に変更）
    filename_jl = replace(filename, r"\.npy$" => ".npz")
    npzwrite(filename_jl, Dict("q_global" => q_global))

    println(@sprintf("qを保存: %s, サイズ=%s", filename_jl, string(size(q_global))))
    return q_global
end

println("共役勾配法（CGM）ソルバー実装完了")