using MuJoCo
using .Threads
using LinearAlgebra

model = load_model("../models/hopper.xml")
data = init_data(model)

function get_state(data)
    return vcat(copy(data.qpos), copy(data.qvel))
end

function RBF(model, data, num_actions, num_obs, num_features, observation; noise_scale=0.1)
    y = zeros(num_features)
    P = randn(num_features, num_obs)
    ν = sqrt(num_obs)
    Φ = 2π .* randn(num_features) .- π

    y .= sin.((P * observation) / ν .+ Φ)

    return y # y is the features that need to be applied in order to actually get an action and policy

end

function rollouts(model, data, policy; N, max_steps)
    trajectories = Vector{Tuple{Vector{Vector{Float64}},Vector{Vector{Float64}},Vector{Float64},Float64}}(undef, N)  # rewards, states, actions

    for n in 1:N

    end
end

function natural_gradient(model, data; K)
    num_actions = model.nu
    observation = get_state(data)
    num_obs = length(observation)
    num_features = 10 * num_obs

    for k = 1:K  # K is the number of iterations that the gradient will perform
        y = RBF(model, data, num_actions, num_obs, num_features, observation)
        W = randn(num_obs, num_features)
        b = randn(num_obs)

        policy = W * y + b

        # collect N trajectories
        # compute the log for the state, action pair
        # compute advantages based on the trajectories; approximate the value function
        # policy gradient




    end

end
