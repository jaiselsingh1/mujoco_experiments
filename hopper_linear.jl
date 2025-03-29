using MuJoCo
using UnicodePlots
using Statistics
using LinearAlgebra

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
num_episodes = 1000
max_steps = 500
ep_rewards = Float64[]


function perturb_state!(data, init_qpos, init_qvel, pertubation_scale = 0.02)
    data.qpos .= copy(init_qpos)
    data.qvel .= copy(init_qvel)
    joint_angles = data.qpos[4:end]
    data.qpos[4:end] .+= pertubation_scale * randn(size(joint_angles))
    data.qvel .+= pertubation_scale * randn(size(data.qvel))
    nothing 
end 

for episode in 1:num_episodes
    global best_policy, best_reward
    policies = []
    rewards = Float64[]
    episode_best_reward = -Inf
    
    for traj in 1:num_trajectories
        perturb_state!(data, init_qpos, init_qvel, 0.02)
        policy = base_policy .+ randn(size(base_policy)).*noise_scale
        push!(policies, policy)
        total_reward = 0.0
        
        for step in 1:max_steps
            observation = vcat(data.qpos, data.qvel)
            action = policy * observation
            data.ctrl .= clamp.(action, -1.0, 1.0)

            step!(model, data)
            upright_bonus = 1.0 
            height = data.qpos[2]
            t_height = 0.85 
            if height > t_height
                total_reward += upright_bonus
                total_reward += data.qvel[1]*4
            else
                total_reward -= t_height-height
            end
            total_reward -= 1e-3 * norm(data.ctrl)^2
            
            #= 
            # COM based reward 
            torso_position = data.xpos[:, 1]
            foot_position = data.xpos[:, 3]

            torso_x = torso_position[1]
            foot_x = foot_position[1]
            balance_reward = -0.2 * abs(torso_x - foot_x)

            energy_penalty = -0.001 * (data.qpos[4]^2 + data.qpos[5]^2 + data.qpos[6]^2 + data.qpos[7]^2)  # penalize any angle deviations for joints 
            target_height = 0.0 
            height_reward = -0.1 * abs(data.qpos[2] - target_height) # data.qpos[2] is rootz 
            velocity_reward = -0.1 * abs(data.qvel[1])
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

