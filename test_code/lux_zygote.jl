using Lux
using Random
using Zygote
using Optimisers
using Plots
using Distributions
using Statistics

# Hyperparameters
N_SAMPLES = 200
LAYERS = [1, 10, 10, 10, 1]
LEARNING_RATE = 0.1
N_EPOCHS = 30_000

function train_model()
    rng = Xoshiro(42)

    # Generate dataset
    x_samples = rand(rng, Uniform(0.0, 2π), (1, N_SAMPLES))
    y_noise = rand(rng, Normal(0.0, 0.3), (1, N_SAMPLES))
    y_samples = sin.(x_samples) .+ y_noise

    # Define model
    model = Chain(
        [Dense(fan_in => fan_out, Lux.sigmoid) for (fan_in, fan_out) in zip(LAYERS[1:end-2], LAYERS[2:end-1])]...,
        Dense(LAYERS[end-1] => LAYERS[end], identity),
    )

    # Initialize parameters and states
    parameters, layer_states = Lux.setup(rng, model)

    # Plot initial predictions
    y_initial_prediction, layer_states = model(x_samples, parameters, layer_states)
    scatter(x_samples[:], y_samples[:], label="data")
    scatter!(x_samples[:], y_initial_prediction[:], label="initial prediction")

    # Loss function
    function loss_fn(p, ls)
        y_pred, new_ls = model(x_samples, p, ls)
        loss = 0.5 * mean((y_pred .- y_samples) .^ 2)
        return loss, new_ls
    end

    # Optimizer setup
    opt = Descent(LEARNING_RATE)
    opt_state = Optimisers.setup(opt, parameters)

    # Training loop
    loss_history = Float64[]
    for epoch in 1:N_EPOCHS
        (loss, layer_states), back = pullback(loss_fn, parameters, layer_states)
        grad, _ = back((1.0, nothing))
        opt_state, parameters = Optimisers.update(opt_state, parameters, grad)

        push!(loss_history, loss)
        if epoch % 100 == 0
            println("Epoch: $epoch, Loss: $loss")
        end
    end

    return parameters, layer_states, loss_history, x_samples, y_samples, model
end


parameters, layer_states, loss_history, x_samples, y_samples, model = train_model()
p = plot(loss_history, yscale=:log10, title="Training Loss", xlabel="Epoch", ylabel="Loss")

y_final_prediction, _ = model(x_samples, parameters, layer_states)
s = scatter(x_samples[:], y_samples[:], label="data")
scatter!(x_samples[:], y_final_prediction[:], label="final prediction")
display(p)
display(s)
readline()



#=
using Lux
using Zygote # automatic diff packages within Julia
using Random
using Optimisers
using Distributions
using Statistics
using Plots

n_samples = 200
layers = [1, 10, 10, 10, 1]
lr = 0.1 # learning rate
num_epochs = 30_000

rng = Xoshiro(42)

x_samples = rand(rng, Uniform(0.0, 2 * π), (1, n_samples))
y_noise = rand(rng, Normal(0.0, 0.3), (1, n_samples))
y_data = sin.(x_samples) .+ y_noise

model = Chain(
    [Dense(fan_in => fan_out, Lux.sigmoid) for (fan_in, fan_out) in zip(layers[1:end-2], layers[2:end-1])]..., # this is a list of layer transitions, so we need to splat it
    Dense(layers[end-1] => layers[end], identity)
)

#lux never stores the parameters, only the arcbitecture
# initialize the parameters (and layer states) -> only relevant if the neural network was stateful
parameters, layer_states = Lux.setup(rng, model)

y_initial_prediction, layer_states = model(x_samples, parameters, layer_states) # have to provide them for Lux
# overwriting the layer_states as they get provided by Lux when you query the model

# the forward function
# maps the parameters to a scalar valued loss function
function loss_fn(parameters, ls)
    y_pred, new_ls = model(x_samples, parameters, ls)
    loss = 0.5 * (y_data .- y_pred) .^ 2
    return loss, new_ls # need to make sure the new layer states are propogated and accessible
end

opt = Descent(lr) #could also use Adam
opt_state = Optimisers.setup(opt, parameters)

# Training loop
loss_history = []
for epoch in 1:num_epochs
    (loss, layer_states), back = pullback(loss_fn, parameters, layer_states) # back does something with cotangents?
    # cotangents are like sensitivity values -> which are then used to compute gradients
    # they are deriatives w/ respect to the output where as the gradients are w/ respect to the input
    gradient, _ = back((1.0, nothing))
    # back is the same signature as the primal output function aka the (loss, new_ls)
    # produces to input of the primal function (params, ls)

    opt_state, parameters = Optimisers.update(opt_state, parameters, gradient)
    push!(loss, loss_history)

    if epoch % 100 == 0
        println("epoch = $epoch, loss =$loss")
    end
end

# primal inputs are the inputs that are provided in the original forward pass in a computational graph

p = plot(loss_history; yscale=:log)
display(p)

=#
