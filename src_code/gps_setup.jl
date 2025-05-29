
using MuJoCo
include("mppi_gps.jl")
using .MPPI

trajectories = load_trajectories("trajectories.jld2")
env = cartpole_model()
init_visualiser()
visualise!(env.model, env.data; trajectories=[traj.states for traj in trajectories])
