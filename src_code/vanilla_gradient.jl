using MuJoCo
using .Threads
using LinearAlgebra

model = load_model("../models/hopper.xml")
data = init_data(model)

function get_state(data)
    return vcat(copy(data.qpos), copy(data.qvel))
end

function hop_reward(data)
    reward = 0.0

    fwd_velocity = data.qvel[1]
    reward += fwd_velocity

    return reward
end

function RBF(model, data)
    num_actions = model.nu
    num_obs = length(get_state(data))
    num_features = 10 * num_obs

    # fixed parammeters
    P = randn(num_features, num_obs)
    # ν = sqrt(num_obs)
    Φ = 2π .* randn(num_features) .- π

    # W_init = randn(num_actions, num_features)
    # b_init = zeros(num_actions)

    function RBF_policy(observation, W, b)
        y = sin.(P * observation + Φ)
        policy = W * y + b
        return policy
    end
    return RBF_policy
end

function collect_trajectories(model, data, policy; N, H, noise_scale=0.1) # N -> number of trajectories ; H -> number of time steps/steps per traj
    trajectories = Vector{Any}(undef, N) # states, actions, rewards
    for n in 1:N
        local_data = init_data(model)

        W = noise_scale * randn(num_actions, num_features)
        b = zeros(num_actions)

        states = Vector{Vector{Float64}}(undef, H)
        actions = Vector{Vector{Float64}}(undef, H)
        rewards = Vector{Float64}(undef, H)

        traj_reward = 0.0
        for h in 1:H
            observation = get_state(local_data)
            states[h] = observation
            actions[h] = policy(observation, W, b)

            local_data.ctrl .= actions[h]
            step!(model, local_data)

            traj_reward = hop_reward(local_data)
            rewards[h] = traj_reward
        end

        trajectories[n] = (states, actions, rewards)
    end

    return trajectories
end

function natural_gradient()
    for k in 1:K

        states, actions, rewards = collect_trajectories(model, data, policy)
    end
end
