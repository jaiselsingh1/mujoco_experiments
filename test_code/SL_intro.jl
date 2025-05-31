using SimpleChains, Random, Optimisers, Zygote, Plots, Statistics

function train_sine(; N=200,
    hidden=16,
    epochs=1_000,
    η=1e-3,
    seed=0)
    Random.seed!(seed)
    x_data = 2π .* rand(Float32, N)
    y_data = sin.(x_data)
    x_train = reshape(x_data, 1, :)
    y_train = reshape(y_data, 1, :)

    chain = SimpleChain(static(1),
        TurboDense(tanh, hidden),
        TurboDense(tanh, hidden),
        TurboDense(identity, 1))

    weights = SimpleChains.init_params(chain, Float32)
    loss(w, x, y) = mean(abs2, chain(x, w) .- y)

    opt = Optimisers.Adam(η)
    state = Optimisers.setup(opt, weights)

    for epoch in 1:epochs
        g = Zygote.gradient(w -> loss(w, x_train, y_train), weights)
        state, weights = Optimisers.update(state, weights, g[1])
        epoch % 100 == 0 && @info "epoch=$epoch  loss=$(loss(weights, x_train, y_train))"
    end

    return chain, weights
end

chain, weights = train_sine()
x_plot = range(0.0f0, 2.0f0 * π; length=200)
y_pred = chain(reshape(Float32.(collect(x_plot)), 1, :), weights)
p = plot(x_plot, sin.(x_plot); label="True sin(x)", lw=2)
plot!(x_plot, vec(y_pred); label="Network", lw=2, ls=:dash)
display(p)
readline()
