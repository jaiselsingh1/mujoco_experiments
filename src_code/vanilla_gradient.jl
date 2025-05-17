using MuJoCo: mjModel, mjData
using MuJoCo
using .Threads
using LinearAlgebra

function get_state(data)
    return vcat(copy(data.qpos), copy(data.qvel))
end

struct HopperModel
    model::mjModel
    data::mjData
    num_actions::Int
    num_obs::Int
    num_features::Int
end

function hopper_model(model_path="../models/hopper.xml")
    model = load_model(model_path)
    data = init_data(model)

    num_actions = model.nu
    num_obs = length(get_state(data))
    num_features = 10 * num_obs

    return HopperModel(model, data, num_actions, num_obs, num_features)
end

function hop_reward(data)
    reward = 0.0

    fwd_velocity = data.qvel[1]
    reward += fwd_velocity

    return reward
end

function RBF(env::HopperModel)
    # fixed parammeters
    P = randn(env.num_features, env.num_obs)
    # ν = sqrt(num_obs)
    Φ = 2π .* randn(env.num_features) .- π

    # W_init = randn(num_actions, num_features)
    # b_init = zeros(num_actions)

    function RBF_policy(observation, W, b)
        y = sin.(P * observation + Φ)
        policy = W * y + b
        return policy
    end
    return RBF_policy
end

function collect_trajectories(env::HopperModel; policy=nothing, N=100, H=1000, noise_scale=0.1) # N -> number of trajectories ; H -> number of time steps/steps per traj
    trajectories = Vector{Any}(undef, N) # states, actions, rewards
    policy = isnothing(policy) ? RBF(env) : policy

    for n in 1:N
        local_data = init_data(env.model)
        W = noise_scale * randn(env.num_actions, env.num_features)
        b = zeros(env.num_actions)

        states = Vector{Vector{Float64}}(undef, H)
        actions = Vector{Vector{Float64}}(undef, H)
        rewards = Vector{Float64}(undef, H)

        traj_reward = 0.0
        for h in 1:H
            observation = get_state(local_data)
            states[h] = observation
            actions[h] = policy(observation, W, b)

            local_data.ctrl .= actions[h]
            step!(env.model, local_data)

            traj_reward = hop_reward(local_data)
            rewards[h] = traj_reward
        end

        trajectories[n] = (states, actions, rewards)
    end

    return trajectories
end
