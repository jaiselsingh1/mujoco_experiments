using SimpleChains, Random, Optimisers, Zygote, Plots, Statistics
using NNlib: relu, sigmoid, leakyrelu

function train_sine(; N=200,
    hidden=128,
    epochs=5_000,
    η=1e-3,
    seed=0,
    activation=tanh,
    tolerance=1e-6)
    Random.seed!(seed)
    x_data = 4π .* rand(Float32, N)
    y_data = 4 .* sin.(x_data) .+ 10
    x_train = reshape(x_data, 1, :)
    y_train = reshape(y_data, 1, :)

    chain = SimpleChain(static(1),
        TurboDense(activation, hidden),
        #TurboDense(activation, hidden),
        TurboDense(identity, 1))

    weights = SimpleChains.init_params(chain, Float32)
    loss(w, x, y) = mean(abs2, chain(x, w) .- y)

    opt = Optimisers.Adam(η)
    state = Optimisers.setup(opt, weights)
    prev_loss = Inf32
    for epoch in 1:epochs
        indices = randperm(N)
        x_shuffled = x_train[:, indices]
        y_shuffled = y_train[:, indices]

        g = Zygote.gradient(w -> loss(w, x_train, y_train), weights) # x train and y train used here for consistency since the shuffle would keep on changing
        state, weights = Optimisers.update(state, weights, g[1])
        epoch % 100 == 0 && @info "epoch=$epoch  loss=$(loss(weights, x_train, y_train))"

        if epoch % 100 == 0
            current_loss = loss(weights, x_shuffled, y_shuffled)
            if abs(prev_loss - current_loss) < tolerance
                @info println("current loss has reached convergence at $epoch")
                break
            end
            prev_loss = current_loss
        end

    end

    return chain, weights
end


function plot_activation(activation)
    chain, weights = train_sine(activation=activation)
    x_plot = range(0.0f0, 6.0f0 * π; length=200)
    y_pred = chain(reshape(Float32.(collect(x_plot)), 1, :), weights)

    p = plot(x_plot, 4 * sin.(x_plot) .+ 10; label="True sin(x)", lw=2)
    plot!(x_plot, vec(y_pred); label="$activation activation", lw=2, ls=:dash)
    display(p)
    readline()
end

activations = [tanh, sin, cos, leakyrelu]
for activation in activations
    plot_activation(activation)
end



#=
@time chain, weights = train_sine()
x_plot = range(0.0f0, 6.0f0 * π; length=200)
y_pred_tanh = chain(reshape(Float32.(collect(x_plot)), 1, :), weights)

chain2, weights2 = train_sine(activation=relu)
y_pred_relu = chain2(reshape(Float32.(collect(x_plot)), 1, :), weights2)

chain3, weights3 = train_sine(activation=x -> (2 * sigmoid(x) - 1))
y_pred_sigmoid = chain3(reshape(Float32.(collect(x_plot)), 1, :), weights3)

chain4, weights4 = train_sine(activation=leakyrelu)
y_pred_leakyrelu = chain4(reshape(Float32.(collect(x_plot)), 1, :), weights4)

p = plot(x_plot, sin.(x_plot); label="True sin(x)", lw=2)
plot!(x_plot, vec(y_pred_tanh); label="tanh", lw=2, ls=:dash)
plot!(x_plot, vec(y_pred_relu); label="relu", lw=2, ls=:dash)
plot!(x_plot, vec(y_pred_sigmoid); label="sigmoid", lw=2, ls=:dash)
plot!(x_plot, vec(y_pred_leakyrelu); label="leaky relu", lw=2, ls=:dash)
=#
