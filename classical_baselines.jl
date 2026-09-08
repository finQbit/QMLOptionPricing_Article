# classical_baselines.jl
#
# Standalone reference implementation of every CLASSICAL model reported in
#
#   "Option Pricing on Noisy Intermediate-Scale Quantum Computers:
#    A Quantum Neural Network Approach"
#
# This file has NO dependency on the authors' internal quantum library. It is
# self-contained and reproduces, from the released datasets, the classical rows
# of the comparison tables: OLS, XGBoost, the parameter-matched MLP, and the
# ridge regression on the quantum model's own Fourier basis.
#
# The quantum model itself is specified in SPECIFICATION.md rather than shipped
# as code, because our implementation sits on an internal library. For a model
# of this size the specification fixes the predictions uniquely, and an
# independent reimplementation is a stronger reproducibility check than running
# our code would be.
#
# Language note: Julia is used here because the quantum-circuit evaluations that
# dominate the study's cost are considerably faster in it than in the equivalent
# Python pipeline. Nothing about the classical baselines requires Julia; they
# translate directly to numpy/scikit-learn.
#
# Dependencies: Distributions, DataFrames, CSV, Random, Statistics,
#               LinearAlgebra, ForwardDiff, XGBoost

using Distributions, DataFrames, CSV, Random, Statistics, LinearAlgebra
using ForwardDiff, XGBoost

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

"""Normalised Black-Scholes-Merton call price, C/K, as a function of
moneyness m = S/K, maturity t, rate r and volatility sigma."""
function bsm_call(m, t, r, sigma)
    d1 = (log.(m) .+ (r .+ sigma .^ 2 / 2) .* t) ./ (sigma .* sqrt.(t))
    d2 = d1 .- sigma .* sqrt.(t)
    return cdf.(Normal(), d1) .* m .- cdf.(Normal(), d2) .* exp.(-r .* t)
end

"""Uniform sampling on the bounds used throughout the paper.

Provided for completeness. To reproduce the published numbers exactly, load the
released CSV files instead: the evaluation set was drawn with Julia's Mersenne
Twister and a reimplementation in another language will not reproduce the same
draw from the same seed."""
function generate_bsm_data(n::Int; rng = Random.default_rng(),
                           lower = [0.8, 0.2, 0.02, 0.01],
                           upper = [1.2, 1.1, 0.10, 1.00])
    m = rand(rng, Uniform(lower[1], upper[1]), n)
    t = rand(rng, Uniform(lower[2], upper[2]), n)
    r = rand(rng, Uniform(lower[3], upper[3]), n)
    s = rand(rng, Uniform(lower[4], upper[4]), n)
    DataFrame(m = m, t = t, r = r, sigma = s, price = bsm_call(m, t, r, s))
end

load_set(path) = CSV.read(path, DataFrame)
features(df)   = Matrix(df[:, [:m, :t, :r, :sigma]])

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

"""MSE, RMSE, MAE and R^2. R^2 uses the standard 1 - SS_res/SS_tot."""
function metrics(y, yhat)
    e = yhat .- y
    mse = mean(e .^ 2)
    (mse = mse, rmse = sqrt(mse), mae = mean(abs.(e)),
     r2 = 1 - sum(e .^ 2) / sum((y .- mean(y)) .^ 2))
end

"""MSE split by moneyness regime, as in the granular comparison tables."""
function metrics_by_regime(m, y, yhat)
    e2 = (yhat .- y) .^ 2
    Dict("OTM" => mean(e2[m .< 0.95]),
         "ATM" => mean(e2[(m .>= 0.95) .& (m .<= 1.05)]),
         "ITM" => mean(e2[m .> 1.05]))
end

"""Error in implied-volatility points via the BSM vega, dIV = dPrice / vega.
Reference point m=1, T=1, r=0.05, sigma=0.2, at which vega = 0.3752."""
function iv_points(err; m = 1.0, t = 1.0, r = 0.05, sigma = 0.2)
    d1 = (log(m) + (r + sigma^2 / 2) * t) / (sigma * sqrt(t))
    vega = m * pdf(Normal(), d1) * sqrt(t)
    return err / vega * 100
end

# ---------------------------------------------------------------------------
# 1. OLS
# ---------------------------------------------------------------------------
# Closed form, therefore deterministic: it carries no seed variance, which is
# why the seed columns of the comparison tables report "--" for this row.

ols_fit(X, y) = hcat(ones(size(X, 1)), X) \ y
ols_predict(beta, X) = hcat(ones(size(X, 1)), X) * beta

# ---------------------------------------------------------------------------
# 2. XGBoost
# ---------------------------------------------------------------------------
# Hyperparameters as stated in the paper. `subsample` and `colsample_bytree`
# make the fit stochastic, hence the seed-to-seed spread reported in the paper
# (R^2 = 0.98459 +/- 0.00190 over seeds {1,2,3,42,123}).

xgb_fit(X, y; seed = 42) =
    xgboost((X, y); num_round = 50, eta = 0.2, max_depth = 3,
            subsample = 0.7, colsample_bytree = 0.8, seed = seed, watchlist = (;))

xgb_predict(model, X) = XGBoost.predict(model, X)

# ---------------------------------------------------------------------------
# 3. Parameter-matched MLP (4-6-1, 37 parameters)
# ---------------------------------------------------------------------------
# Matched to the quantum model's 36 parameters. Inputs are rescaled to [-1,1]
# using the sampling bounds; tanh hidden layer, linear output; Adam.

const MLP_LB = [0.8, 0.2, 0.02, 0.01]
const MLP_UB = [1.2, 1.1, 0.10, 1.00]

"""Rescale a 4 x N feature matrix to [-1,1] using the sampling bounds."""
rescale(X) = 2.0 .* (X .- MLP_LB) ./ (MLP_UB .- MLP_LB) .- 1.0

function mlp_unpack(theta)
    (reshape(theta[1:24], 6, 4), theta[25:30], reshape(theta[31:36], 1, 6), theta[37])
end

function mlp_forward(theta, Xn)
    W1, b1, W2, b2 = mlp_unpack(theta)
    vec(W2 * tanh.(W1 * Xn .+ b1) .+ b2)
end

mlp_loss(theta, Xn, y) = mean((mlp_forward(theta, Xn) .- y) .^ 2)

"""Train the MLP with Adam. Initialisation N(0, 0.5^2); 2000 iterations at
learning rate 0.03, as reported."""
function mlp_train(Xn, y; seed = 1, iters = 2000, lr = 0.03)
    rng = MersenneTwister(seed)
    theta = randn(rng, 37) .* 0.5
    m = zeros(37); v = zeros(37)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for k in 1:iters
        g = ForwardDiff.gradient(th -> mlp_loss(th, Xn, y), theta)
        m .= b1 .* m .+ (1 - b1) .* g
        v .= b2 .* v .+ (1 - b2) .* (g .^ 2)
        theta .-= lr .* (m ./ (1 - b1^k)) ./ (sqrt.(v ./ (1 - b2^k)) .+ eps)
    end
    return theta
end

# ---------------------------------------------------------------------------
# 4. Ridge regression on the quantum model's own Fourier basis
# ---------------------------------------------------------------------------
# This is the decisive test requested in review. Following Schuld, Sweke and
# Meyer, the re-uploading circuit is a truncated Fourier series whose frequency
# support is fixed by the encoding generators. Each of the four features is
# encoded L = 3 times, so the accessible per-feature range is k in {1,2,3}.
#
# The basis is built on the RESCALED features. This matters: on raw inputs,
# cos(k*r) with r ~ 0.05 is nearly constant and the basis degenerates.
#
# PAIRS are the feature pairs that share a qubit within a single layer, and are
# therefore directly entangled by the CX block acting on that pair. They are
# read off the encoding schedule, not chosen: (m,sigma), (t,r), (m,t), (r,sigma).

const PAIRS = [(1, 4), (2, 3), (1, 2), (3, 4)]   # features ordered (m, t, r, sigma)

"""Additive basis: intercept plus {cos(k x_i), sin(k x_i)} for k = 1..L.
With L = 3 and four features this gives 1 + 4*3*2 = 25 columns."""
function fourier_additive(Xn; L = 3)
    n = size(Xn, 2)
    cols = [ones(n)]
    for i in 1:4, k in 1:L
        push!(cols, cos.(k .* Xn[i, :]))
        push!(cols, sin.(k .* Xn[i, :]))
    end
    cols
end

"""Cross terms for the entangled pairs, harmonics 1..L.
L = 1 adds 4*4 = 16 columns (variant B, 41 total);
L = 2 adds 4*4*4 = 64 columns (variant C, 89 total)."""
function fourier_cross(Xn; L = 1)
    cols = Vector{Float64}[]
    for (i, j) in PAIRS, ki in 1:L, kj in 1:L
        xi, xj = Xn[i, :], Xn[j, :]
        push!(cols, cos.(ki .* xi) .* cos.(kj .* xj))
        push!(cols, cos.(ki .* xi) .* sin.(kj .* xj))
        push!(cols, sin.(ki .* xi) .* cos.(kj .* xj))
        push!(cols, sin.(ki .* xi) .* sin.(kj .* xj))
    end
    cols
end

"""Select the ridge penalty by k-fold cross-validation on the training set.

The grid is deliberately bounded below at 1e-2. Beneath that the Gram matrix of
correlated trigonometric features is near-singular, the cross-validation surface
is very flat, and the selected optimum shifts between runs by up to 0.001 in R^2
under different floating-point summation orders. The reported ceiling is stable
across the retained range; a sweep over the whole grid is given in the paper."""
function cv_ridge(F, y; k = 5, lambdas = 10.0 .^ (-2:0.25:2), seed = 1)
    n = size(F, 1)
    idx = shuffle(MersenneTwister(seed), 1:n)
    fold = div(n, k)
    best_lam, best = lambdas[1], Inf
    for lam in lambdas
        errs = Float64[]
        for f in 1:k
            te = idx[((f - 1) * fold + 1):(f == k ? n : f * fold)]
            tr = setdiff(idx, te)
            A = F[tr, :]
            beta = (A' * A + lam * I(size(F, 2))) \ (A' * y[tr])
            push!(errs, mean((F[te, :] * beta .- y[te]) .^ 2))
        end
        e = mean(errs)
        e < best && (best_lam, best = lam, e)
    end
    return best_lam
end

function ridge_fit_predict(cols_tr, cols_te, ytr; lambda = nothing)
    Ftr, Fte = hcat(cols_tr...), hcat(cols_te...)
    lam = lambda === nothing ? cv_ridge(Ftr, ytr) : lambda
    beta = (Ftr' * Ftr + lam * I(size(Ftr, 2))) \ (Ftr' * ytr)
    return (pred = Fte * beta, lambda = lam, npar = size(Ftr, 2))
end

# ---------------------------------------------------------------------------
# Worked example
# ---------------------------------------------------------------------------
# Reproduces the classical rows of the main comparison table. Expected values,
# on the released evaluation set:
#
#   OLS                       R^2 = 0.96154   MAE = 0.01626
#   Fourier ridge (B), 41 par R^2 = 0.97212   MSE = 0.00036
#   XGBoost (seed 42)         R^2 = 0.98641   MAE = 0.01043
#   XGBoost (5 seeds)         R^2 = 0.98459 +/- 0.00190
#   MLP (seed 1)              R^2 = 0.99266   MAE = 0.00692
#   MLP (5 seeds)             R^2 = 0.99604 +/- 0.00203
#
# For reference, the quantum model scores R^2 = 0.98721, MAE = 0.00965 on the
# same set.

function reproduce(; traindir = "data", evalfile = "data/bs_eval_10000.csv")
    tr = load_set(joinpath(traindir, "bs_train.csv"))
    isfile(evalfile) || error("""
        Missing $evalfile.
        Export it once from the generating environment, or substitute
        bs_test.csv to run against the smaller N=100 set.""")
    te = load_set(evalfile)
    Xtr, ytr = features(tr), tr.price
    Xte, yte = features(te), te.price
    Xtr_n, Xte_n = rescale(Xtr'), rescale(Xte')

    println("model                       R^2        MAE        MSE")
    println("-"^56)

    b = ols_fit(Xtr, ytr); m1 = metrics(yte, ols_predict(b, Xte))
    println(rpad("OLS", 26), rpad(round(m1.r2, digits=5), 11),
            rpad(round(m1.mae, digits=5), 11), round(m1.mse, digits=5))

    rb = ridge_fit_predict(vcat(fourier_additive(Xtr_n), fourier_cross(Xtr_n; L=1)),
                           vcat(fourier_additive(Xte_n), fourier_cross(Xte_n; L=1)), ytr)
    m2 = metrics(yte, rb.pred)
    println(rpad("Fourier ridge (B, $(rb.npar)p)", 26), rpad(round(m2.r2, digits=5), 11),
            rpad(round(m2.mae, digits=5), 11), round(m2.mse, digits=5))

    for s in [1, 2, 3, 42, 123]
        mx = metrics(yte, xgb_predict(xgb_fit(Xtr, ytr; seed=s), Xte))
        println(rpad("XGBoost (seed $s)", 26), rpad(round(mx.r2, digits=5), 11),
                rpad(round(mx.mae, digits=5), 11), round(mx.mse, digits=5))
    end

    for s in [1, 2, 3, 42, 123]
        mm = metrics(yte, mlp_forward(mlp_train(Xtr_n, ytr; seed=s), Xte_n))
        println(rpad("MLP (seed $s)", 26), rpad(round(mm.r2, digits=5), 11),
                rpad(round(mm.mae, digits=5), 11), round(mm.mse, digits=5))
    end
    return nothing
end
