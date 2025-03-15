using MuJoCo 
using Statistics 
using LinearAlgebra 
using UnicodePlots 

model = load_model("pointmass.xml")
data = init_data(model)

num_observations = 2*model.nq   # this can be any size but ideally it can be of any size like an image has everything 
num_actions = model.nu 
num_features = 10*num_observations 

mj_resetData(model, data)
init_qpos = copy(data.qpos)
init_qvel = copy(data.qvel)

bandwidth = 0.5 
global W = randn(num_features, num_observations) * bandwidth
global b = rand(num_features) .* 2π .- π
base_policy = 0.0 * randn(num_actions, num_features)

global best_reward = -Inf #common to use -Inf when you're trying to max any value 
global best_policy = copy(base_policy)
global best_total_reward = -Inf 

num_trajectories = 2*length(base_policy) # based on simplex methods or finite differencing 
# you can have n samples for a policy of a specific length 
num_episodes = 100 
max_steps = 1000 
noise_scale = 0.05
learning_rate = 0.2

ep_rewards = Float64[]
for episode in 1:num_episodes
    global base_policy, best_policy, best_reward, best_total_reward, W, b 
    policies = []
    rewards = Float64[]
    episode_best_reward = -Inf 

    for i in 1:num_trajectories 

        policy = base_policy .+ randn(size(base_policy))*noise_scale
        push!(policies, policy)

        data.qpos .= init_qpos # making sure that the data is reset without calling mj_resetData()
        data.qvel .= init_qvel

        # random placement to begin 
        data.qpos[1] = 0.2 * (rand() - 0.5) 
        data.qpos[2] = 0.2 * (rand() - 0.5)

        # reward per trajectory 
        total_reward = 0.0 

        for j in 1:max_steps
        
            state = vcat(data.qpos, data.qvel)
            observation = sin.(W * state .+ b)

            action = policy * observation #linear policy 
            data.ctrl .= clamp.(action, -1.0, 1.0)
            step!(model, data)

            x = data.qpos[1]
            y = data.qpos[2]

            dist_target = sqrt(x^2 + y^2)
            position_reward = -dist_target
            velocity_penalty = 0.01 * (data.qvel[1]^2 + data.qvel[2]^2)
            control_penalty = 0.005 * sum(abs.(data.ctrl))
            step_reward = position_reward # - velocity_penalty - control_penalty 

            total_reward += step_reward 
        end 

        if total_reward > mean(best_total_reward)
            best_total_reward = total_reward 
            best_policy = copy(policy)
            println("New best policy found Reward: $best_total_reward")
        end 

        if total_reward > episode_best_reward
            episode_best_reward = total_reward 
        end 

        push!(rewards, episode_best_reward)
    end 

    normalized_rewards = (rewards .- mean(rewards)) ./ (std(rewards) + 1e-8)
    push!(ep_rewards, mean(rewards)) #mean from the reward gathered in the trajectory

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
        println("Episode $episode | Avg Reward: $(mean(rewards)) | Best Episode: $episode_best_reward | All-time Best: $best_total_reward")
    end
end 

function trained_policy_controller!(m::Model, d::Data)
    state = vcat(d.qpos, d.qvel)
    observation = sin.(W * state .+ b)
    d.ctrl .= clamp.(best_policy * observation, -1.0, 1.0)
    nothing
end


mj_resetData(model, data)
data.qpos[1] = 0.2  # random initial poisitions for test 
data.qpos[2] = 0.2 
init_visualiser()
visualise!(model, data, controller = trained_policy_controller!)




