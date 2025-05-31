
using MuJoCo
using SimpleChains
include("mppi_gps.jl")
using .MPPI

trajectories = load_trajectories("trajectories.jld2")
env = cartpole_model()
# init_visualiser()
# visualise!(env.model, env.data; trajectories=[traj.states for traj in trajectories])

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

function train_policy!(policy, states, actions; epochs=1000)
    weights = SimpleChains.init_params(policy, Float32)


end
