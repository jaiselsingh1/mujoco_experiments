using MuJoCo
using SimpleChains, Random, Optimisers, Zygote, Plots, Statistics
include("mppi_gps.jl")
using .MPPI

function policy_network(env; activation=tanh, hidden=128)
    state_dim = env.state_dim
    action_dim = env.ν
    policy = SimpleChain(static(state_dim),
        TurboDense(activation, hidden),
        TurboDense(activation, hidden),
        TurboDense(identity, action_dim))
    return policy
end

function extract_trajectories(trajectories)
    all_states = []
    all_actions = []

    for traj in trajectories
        push!(all_states, traj.states)
        push!(all_actions, traj.controls)
    end

    states = hcat(all_states...)
    actions = hcat(all_actions...)
    return states, actions
end

function train_policy!(policy, trajectories; epochs=1000, lr=0.01, batch_size=25)
    weights = SimpleChains.init_params(policy, Float32)
    loss(weights, states, actions) = mean(abs2, policy(states, weights) .- actions)

    #these are the dataset that we are doing BC on
    states, actions = extract_trajectories(trajectories)

    num_samples = size(states, 2) # 4 states for 100, time steps hence 100 samples

    opt = Optimisers.Adam(lr)
    opt_state = Optimisers.setup(opt, weights)

    for epoch in 1:epochs
        indices = randperm(num_samples)
        states_shuffled = states[:, indices]
        actions_shuffled = actions[:, indices] # still 4x1000 size

        # introduce batching
        for i in 1:batch_size:num_samples
            end_idx = min(i + batch_size - 1, num_samples)

            states_batch = states_shuffled[:, i:end_idx]
            actions_batch = actions_shuffled[:, i:end_idx]
            g = Zygote.gradient(w -> loss(w, states_batch, actions_batch), weights)
            opt_state, weights = Optimisers.update(opt_state, weights, g[1])
        end

        if epoch % 100 == 0
            current_loss = loss(weights, states, actions)
            @info "epoch=$epoch  loss=$current_loss"
        end
    end
    return policy, weights
end


function policy_guided_mppi(env, policy, weights, planner::MPPIPlanner; T=100)
    model = env.model
    data = env.data

    # temp data to make sure you're not affecting the actual data
    temp_data = init_data(model)
    temp_data.qpos .= data.qpos
    temp_data.qvel .= data.qvel

    for t in 1:T
        state = get_physics_state(model, temp_data)
        action = policy(state, weights)

        planner.U[:, t] .= action
        temp_data.ctrl .= action
        step!(model, temp_data)
    end

    mppi_traj!(env, planner)

    return planner
end



starting_states, trajectories = load_trajectories("trajectories.jld2")
env = cartpole_model()
model, data = env.model, env.data
policy = policy_network(env)


policy, weights = train_policy!(policy, trajectories)
function trained_policy_controller!(model, data)
    state = get_physics_state(model, data)
    data.ctrl .= policy(state, weights)
    nothing
end


reset!(model, data)
init_visualiser()
visualise!(model, data, controller=trained_policy_controller!)
