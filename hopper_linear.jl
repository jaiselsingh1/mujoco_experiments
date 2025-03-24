using MuJoCo
using UnicodePlots
using Statistics

model = load_model("hopper.xml")
data = init_data(model)
# model.nq = 7
# model.nu = 4
mj_resetData(model, data)
init_qpos = copy(data.qpos) # to use within the loop to reset the data
init_qvel = copy(data.qvel)
num_observations = 2*model.nq
num_actions = model.nu
noise_scale = 0.05
learning_rate = 0.2
base_policy = 0.0 * randn(num_actions, num_observations)
global best_policy = copy(base_policy)
global best_reward = -Inf
num_trajectories = 2*length(base_policy)
num_episodes = 500
max_steps = 500
ep_rewards = Float64[]

for episode in 1:num_episodes
    global best_policy, best_reward
    policies = []
    rewards = Float64[]
    episode_best_reward = -Inf
    
    for traj in 1:num_trajectories
        data.qpos .= copy(init_qpos) # instead of mj reset data
        data.qvel .= copy(init_qvel)
        policy = base_policy .+ randn(size(base_policy)).*noise_scale
        push!(policies, policy)
        total_reward = 0.0
        
        for step in 1:max_steps
            observation = vcat(data.qpos, data.qvel)
            action = policy * observation
            data.ctrl .= clamp.(action, -1.0, 1.0)

            step!(model, data)
            alive_bonus = 1.0 
            upright_reward = 1.0 - abs(data.qpos[3])  # higher reward when angle is closer to zero
            target_height = 1.0 
            height_reward = 1.0 - abs(data.qpos[2] - target_height)
            stay_still_reward = -0.1 * abs(data.qvel[1])  # penalize horizontal velocity
            energy_penalty = -0.01 * sum(x->x^2, data.ctrl)

            total_reward += alive_bonus + upright_reward + height_reward + stay_still_reward - energy_penalty

            #=
            alive_bonus = 1.0
            total_reward += data.qvel[1]
            total_reward += alive_bonus
            total_reward -= 1e-3 * sum(x->x^2, action) # summing the square of each action component
            =#

        end
        
        if total_reward > best_reward
            best_reward = total_reward
            best_policy = copy(policy)
            println("New best policy found! Reward: $best_reward")
        end
        
        if total_reward > episode_best_reward
            episode_best_reward = total_reward
        end
        
        push!(rewards, total_reward)
    end
    
    normalized_rewards = (rewards .- mean(rewards)) ./ (std(rewards) + 1e-8)
    push!(ep_rewards, mean(rewards))
    
    gradient = zeros(size(base_policy))
    for i in 1:num_trajectories
        noise = policies[i] - base_policy
        gradient .+= noise .* normalized_rewards[i]
    end
    
    display(heatmap(base_policy))
    display(gradient)
    display(lineplot(ep_rewards))
    
    base_policy .+= learning_rate * gradient/num_trajectories
    
    if episode % 5 == 0 || episode == 1
        println("Episode $episode | Avg Reward: $(mean(rewards)) | Best Episode: $episode_best_reward | All-time Best: $best_reward")
    end
end

function trained_policy_controller!(m::Model, d::Data)
    state = vcat(d.qpos, d.qvel)
    d.ctrl .= clamp.(best_policy * state, -1.0, 1.0)
    nothing
end

mj_resetData(model, data)
init_visualiser()
visualise!(model, data, controller = trained_policy_controller!)

