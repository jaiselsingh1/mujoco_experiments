using MuJoCo

const torque_bins = [-8.0, 0.0, 8.0]
const ϵ_start = 0.2
const ϵ_decay = 0.995
const α             = 0.1 #learning rate
const γ             = 0.99   #discount factor
const episodes      = 8_000
const T             = 200 #steps per rollout
const angle_bins      = collect(range(-π,  π;  length=31))  # θ
const vel_bins      = collect(range(-10, 10;  length=31)) # θ̇_dot

const ns_a = length(angle_bins)
const ns_v = length(vel_bins)
const na = length(torque_bins)
const Q = zeros(Float32, ns_a, ns_v, na)

function angle_idx(theta)
    clamp(searchsortedfirst(angle_bins, theta), 1, ns_a)
end

function vel_idx(theta_dot)
    clamp(searchsortedfirst(vel_bins, theta_dot), 1, ns_v)
end

m = load_model("../models/cartpole.xml")
d = init_data(m)



for ep in 1:episodes
   reset!(m, d)
   ϵ = ϵ_start
   θ, θ̇_dot = d.qpos[2], d.qvel[2]
   ia, iv = angle_idx(θ), vel_idx(θ̇_dot)

   for t in 1:T
       a_idx = rand() < ϵ ? rand(1:na) : argmax(Q[ia, iv, :])
       d.ctrl[1] = torque_bins[a_idx]

       step!(m, d)
       θ_new, θ̇_new = d.qpos[2], d.qvel[2]
       ia2, iv2 = angle_idx(θ_new), vel_idx(θ̇_new)

       reward = 1.0 - abs(θ_new) / π

       Q[ia, iv, a_idx] += α * (reward + γ * maximum(Q[ia2, iv2, :]) - Q[ia, iv, a_idx])

       ia, iv = ia2, iv2
   end

   ϵ *= ϵ_decay

   if ep % 500 == 0
       println("Episode $ep, ε = $(round(ϵ, digits=3))")
   end
end

function greedy_policy!(m, d)
   θ, θ̇_dot = d.qpos[2], d.qvel[2]
   ia, iv = angle_idx(θ), vel_idx(θ̇_dot)
   best_a = argmax(Q[ia, iv, :])
   d.ctrl .= torque_bins[best_a]
end

reset!(m, d)
init_visualiser()
visualise!(m, d; controller=greedy_policy!)
