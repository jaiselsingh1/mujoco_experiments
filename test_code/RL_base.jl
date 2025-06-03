using MuJoCo
using UnicodePlots

const force_bins = collect(range(-1.0, 1.0; length=3))
const ϵ_start = 0.3
const ϵ_decay = 0.999
const α = 0.2
const γ = 0.99
const episodes = 10_000
const T = 300

const x_bins = collect(range(-2.4, 2.4; length=11))
const x_dot_bins = collect(range(-4, 4; length=11))
const angle_bins = collect(range(-π/4, π/4; length=11))
const vel_bins = collect(range(-2, 2; length=11))

const ns_x = length(x_bins)
const ns_xd = length(x_dot_bins)
const ns_a = length(angle_bins)
const ns_v = length(vel_bins)
const na = length(force_bins)

function x_idx(x)
    clamp(searchsortedfirst(x_bins, x), 1, ns_x)
end

function x_dot_idx(x_dot)
    clamp(searchsortedfirst(x_dot_bins, x_dot), 1, ns_xd)
end

function angle_idx(theta)
    clamp(searchsortedfirst(angle_bins, theta), 1, ns_a)
end

function vel_idx(theta_dot)
    clamp(searchsortedfirst(vel_bins, theta_dot), 1, ns_v)
end

function train_q_learning()
    Q = zeros(Float32, ns_x, ns_xd, ns_a, ns_v, na)
    episode_rewards = Float64[]
    ϵ = ϵ_start

    for ep in 1:episodes
        reset!(m, d)
        ep_reward = 0.0

        x, θ = d.qpos[1], d.qpos[2]
        x_dot, θ_dot = d.qvel[1], d.qvel[2]
        ix, ixd, ia, iv = x_idx(x), x_dot_idx(x_dot), angle_idx(θ), vel_idx(θ_dot)

        for t in 1:T
            # off-policy TD learning
            a_idx = rand() < ϵ ? rand(1:na) : argmax(Q[ix, ixd, ia, iv, :])

            d.ctrl .= force_bins[a_idx]
            step!(m, d)

            x_new, θ_new = d.qpos[1], d.qpos[2]
            x_dot_new, θ_dot_new = d.qvel[1], d.qvel[2]
            ix2, ixd2, ia2, iv2 = x_idx(x_new), x_dot_idx(x_dot_new), angle_idx(θ_new), vel_idx(θ_dot_new)

            angle_reward = cos(θ_new)
            pos_penalty = 0.1 * abs(x_new)
            angle_vel_penalty = 0.05 * abs(θ_dot_new)
            pos_vel_penalty = 0.05 * abs(x_dot_new)
            reward = angle_reward - pos_penalty - angle_vel_penalty - pos_vel_penalty
            ep_reward += reward

            Q[ix, ixd, ia, iv, a_idx] += α * (reward + γ * maximum(Q[ix2, ixd2, ia2, iv2, :]) - Q[ix, ixd, ia, iv, a_idx])
            ix, ixd, ia, iv = ix2, ixd2, ia2, iv2
        end

        push!(episode_rewards, ep_reward)
        ϵ *= ϵ_decay
    end

    display(lineplot(episode_rewards, title="Episode Rewards"))

    return Q, m, d
end

function greedy_policy!(m, d)
    x, θ = d.qpos[1], d.qpos[2]
    x_dot, θ_dot = d.qvel[1], d.qvel[2]

    ix, ixd, ia, iv = x_idx(x), x_dot_idx(x_dot), angle_idx(θ), vel_idx(θ_dot)
    best_a = argmax(Q[ix, ixd, ia, iv, :])
    d.ctrl .= force_bins[best_a]
end


m = load_model("../models/cartpole.xml")
d = init_data(m)
Q, m, d = train_q_learning()
reset!(m, d)
init_visualiser()
visualise!(m, d; controller=greedy_policy!)
