using MuJoCo
using UnicodePlots

const torque_bins = collect(range(-1.0, 1.0; length=3))
const ϵ_start = 0.3
const ϵ_decay = 0.999
const α = 0.3
const γ = 0.99
const episodes = 8_000
const T = 300

const angle_bins = collect(range(-π, π; length=51))
const vel_bins = collect(range(-6, 6; length=51)) # plotting the observed velocities

const ns_a = length(angle_bins) # number of angle bins
const ns_v = length(vel_bins) # number of velocity bins
const na = length(torque_bins) # number of torque bins

function angle_idx(theta)
    clamp(searchsortedfirst(angle_bins, theta), 1, ns_a)
end

function vel_idx(theta_dot)
    clamp(searchsortedfirst(vel_bins, theta_dot), 1, ns_v)
end

function train_q_learning()
    Q = zeros(Float32, ns_a, ns_v, na)
    m = load_model("../models/cartpole.xml")
    d = init_data(m)
    episode_rewards = Float64[]
    obs_vels = Float64[]

    ϵ = ϵ_start
    for ep in 1:episodes
        reset!(m, d)
        θ, θ̇_dot = d.qpos[2], d.qvel[2]
        ia, iv = angle_idx(θ), vel_idx(θ̇_dot)

        for t in 1:T
            a_idx = rand() < ϵ ? rand(1:na) : argmax(Q[ia, iv, :])   # argmax is looking at the torque bins for the best action for that given state pair
            d.ctrl[1] = torque_bins[a_idx]

            step!(m, d)
            θ_new, θ̇_dot_new = d.qpos[2], d.qvel[2]
            push!(obs_vels, θ̇_dot_new)
            ia2, iv2 = angle_idx(θ_new), vel_idx(θ̇_dot_new)

            reward = abs(θ_new) <= π/2 ? 3.0 - abs(θ_new)/(π/2) : -1.0
            # reward = 1.0 - abs(θ_new) / π
            push!(episode_rewards, reward)
            Q[ia, iv, a_idx] += α * (reward + γ * maximum(Q[ia2, iv2, :]) - Q[ia, iv, a_idx])

            ia, iv = ia2, iv2
        end

        ϵ *= ϵ_decay
        if ep % 500 == 0
            println("Episode $ep, ε = $(round(ϵ, digits=3))")
        end
    end

    display(lineplot(episode_rewards))
    display(histogram(obs_vels))
    return Q, m, d
end

Q, m, d = train_q_learning()


function greedy_policy!(m, d)
    θ, θ̇_dot = d.qpos[2], d.qvel[2]
    ia, iv = angle_idx(θ), vel_idx(θ̇_dot)
    best_a = argmax(Q[ia, iv, :])
    d.ctrl[1] = torque_bins[best_a]
end

reset!(m, d)
init_visualiser()
visualise!(m, d; controller=greedy_policy!)
