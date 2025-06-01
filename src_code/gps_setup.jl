using MuJoCo
using SimpleChains, Random, Optimisers, Zygote, Plots, Statistics
include("mppi_gps.jl")
using .MPPI

function policy_network(env)
    state_dim = env.state_dim
    action_dim = env.ν
    policy = SimpleChain(static(state_dim),
                        TurboDense(tanh, 32),
                        TurboDense(tanh, 16),
                        TurboDense(identity, action_dim))
    return policy
end

function extract_trajectories(trajectories)
    all_states = []
    all_actions = []

    for traj in trajectories
        for i in 1:(length(traj.states))
            push!(all_states, traj.states[i])
            push!(all_actions, traj.controls[i])
        end
    end
    states = hcat(all_states...)
    actions = hcat(all_actions...)
    return states, actions
end

function train_policy!(policy, trajectories; epochs=1000, lr=0.001)
    weights = SimpleChains.init_params(policy, Float32)
    loss(weights, states, actions) = mean(abs2, policy(states, weights) .- actions)

    states, actions = extract_trajectories(trajectories)
    num_samples = size(states, 2)

    opt = Optimisers.Adam(lr)
    opt_state = Optimisers.setup(opt, weights)

    for epoch in 1:epochs
        indices = randperm(num_samples)
        states_shuffled = states[:, indices]
        actions_shuffled = actions[:, indices]

        g = Zygote.gradient(w -> loss(w, states_shuffled, actions_shuffled), weights)
        opt_state, weights = Optimisers.update(opt_state, weights, g[1])

        if epoch % 100 == 0
            current_loss = loss(weights, states, actions)
            @info "epoch=$epoch  loss=$current_loss"
        end
    end

    return policy, weights
end


trajectories = load_trajectories("trajectories.jld2")
env = cartpole_model()
model, data = env.model, env.data
policy = policy_network(env)


policy, weights = train_policy!(policy, trajectories)

function create_trained_controller(policy, weights)
    return function trained_policy_controller!(model, data)
        state = get_physics_state(model, data)
        data.ctrl .= policy(state, weights)
        nothing
    end
end

controller = create_trained_controller(policy, weights)
mj_resetData(model, data)
init_visualiser()
visualise!(model, data, controller=controller)



#=
using MuJoCo
using SimpleChains, Random, Optimisers, Zygote, Plots, Statistics
include("mppi_gps.jl")
using .MPPI


function policy_network(env) # not linear with the tanh
    state_dim = env.state_dim
    action_dim = env.ν

    policy = SimpleChain(static(state_dim), TurboDense(tanh, 32), TurboDense(tanh, 16), TurboDense(identity, action_dim))
    return policy
end

function extract_trajectories(trajectories)
    all_states = []
    all_actions = []

    for traj in trajectories
        for i in (length(traj.states) - 1)
            all_states = [traj.states[i] for traj in trajectories]
            all_actions = [traj.controls[i] for traj in trajectories]
        end
    end

    states = hcat(all_states...)
    actions = hcat(all_actions...) # the ... is a splat operator
    return states, actions
end

function train_policy!(policy; epochs=1000, lr=0.001)
    weights = SimpleChains.init_params(policy, Float32)

    loss(weights, states, actions) = mean(abs2, policy(states, weights) .- actions)

    states, actions = extract_trajectories(trajectories)

    opt = Optimisers.Adam(lr)
    state = Optimisers.setup(opt, weights)
    for epoch in 1:epochs
        indices = randperm(length(states))
        states_shuffed = states[:, indices]
        actions_shuffled = actions[:, indices]

        g = Zygote.gradient(weights -> loss(weights, states_shuffled, actions_shuffled), weights)
        state, weights = Optimisers.update(states, weights, g[1])
        epoch % 100 == 0 && @info "epoch=$epoch  loss=$(loss(weights, states, actions))"
    end

    return policy, weights
end

trajectories = load_trajectories("trajectories.jld2")
env = cartpole_model()
model, data = env.model, env.data

policy = policy_network(env)
policy, weights = train_policy!(policy)

function trained_policy_controller!(model, data)
    state = get_physics_state(model, data)
    data.ctrl .= policy * state
    nothing
end

mj_resetData(model, data)
init_visualiser()
visualise!(model, data, controller=trained_policy_controller!)

# init_visualiser()
# visualise!(env.model, env.data; trajectories=[traj.states for traj in trajectories])

=#
