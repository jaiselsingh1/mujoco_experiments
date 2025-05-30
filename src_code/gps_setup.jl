
using MuJoCo
using SimpleChains
include("mppi_gps.jl")
using .MPPI

trajectories = load_trajectories("trajectories.jld2")
env = cartpole_model()
# init_visualiser()
# visualise!(env.model, env.data; trajectories=[traj.states for traj in trajectories])

function linear_policy(env)
    state_dim = env.state_dim
    action_dim = env.ν

    policy = SimpleChain(static(state_dim), TurboDense(tanh, 32), TurboDense(tanh, 16), TurboDense(identity, action_dim))
    return policy
end

function extract_trajectories(trajectories)
    all_states = [traj.states for traj in trajectories]
    all_actions = [traj.controls for traj in trajectories]

    return all_states, all_actions
end
