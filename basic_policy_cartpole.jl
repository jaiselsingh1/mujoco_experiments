using MuJoCo 
using Flux 
using LinearAlgebra 
using Statistics

model = load_model("cartpole.xml")
data = init_data(model)
num_observations = 2*model.nq # number of observable states 
num_actions = model.nu # number of actuators 

base_policy = 0.01 * randn(num_actions, num_observations) # not starting with zeros 
global best_reward = -Inf
global best_policy = copy(base_policy)
global best_total_reward = -Inf  # best total trajectory reward 

num_trajectories = 10 
num_episodes = 100 # total training episodes 
max_steps = 1000 # maximum steps per trajectory 
noise_scale = 0.1 # for policy updates
learning_rate = 0.02

# outermost is training loop -> policy parameters change per steps
# outer for loop -> sampling policies 
# inner for loop ->  using policies to get trajectories

# Training loop is outermost where the policy params change through steps. 
# Outer loop is sampling policies. Inter loop is using policy to get trajectories

for episode in 1:num_episodes
    global base_policy, best_policy, best_reward, best_total_reward 
    policies = []
    rewards = [] 
    episode_best_reward = -Inf

    for i in 1:num_trajectories
        # get one or more trajectories and their reward -> add noise every time to the policy 
        mj_resetData(model, data)
        policy = base_policy .+ randn(size(base_policy)) * noise_scale
        push!(policies, policy)

        total_reward = 0.0
        trajectory_rewards = []

        for step in 1:max_steps 
            # get reward/ observations (multiply with policy to get actions -> apply actions to sample_model_and_data)
            # step forward in time (get trajectory/ get reward)
            observation = vcat(data.qpos, data.qvel) 
            if size(observation, 2) != 1
                observation = reshape(observation, :, 1) # make it into a column vector 
            end 

            action = policy * observation #simple linear policy

            data.ctrl .= clamp.(action, -1.0, 1.0)
        
            mj_step(model, data)

            pole_angle = data.qpos[2]
            cart_pos = data.qpos[1] 
            
            angle_reward = cos(pole_angle) # straight angle is better hence reward for the theta to be close to 0 
            pos_penalty = 0.1 * abs(cart_pos)  # Small penalty for distance from center
            
            angle_vel_penalty = 0.05 * abs(data.qvel[2])  # Penalize fast angle changes
            pos_vel_penalty = 0.05 * abs(data.qvel[1])    # Penalize fast cart movement
            
            # Combined reward
            step_reward = angle_reward - pos_penalty - angle_vel_penalty - pos_vel_penalty
            
            total_reward += step_reward 
            
            if step == max_steps
                total_reward += 10.0 # to reach the last step give a reward  
            end 
        end 
        push!(rewards, total_reward)

        if total_reward > best_total_reward
            best_total_reward = total_reward 
            best_policy = copy(policy)
            println("New best policy found Reward: $best_total_reward")
        end 

        if total_reward > episode_best_reward
            episode_best_reward = total_reward 
        end 

    end 
    

    #combine -> weight all the policies with the rewards generated to approximate a gradient direction and take a step 
    # training loop 
    normalized_rewards = (rewards .- mean(rewards)) ./ (std(rewards) + 1e-8)
    

    gradient = zeros(size(base_policy))
    for i in 1:num_trajectories
        noise = policies[i] - base_policy 
        gradient .+= noise .* normalized_rewards[i]
    end 
    
    base_policy .+= learning_rate * gradient/num_trajectories 

    if episode % 5 == 0 || episode == 1
        println("Episode $episode | Avg Reward: $(mean(rewards)) | Best Episode: $episode_best_reward | All-time Best: $best_total_reward")
    end
end 

function trained_policy_controller!(m::Model, d::Data)
    state = vcat(d.qpos, d.qvel)
    d.ctrl .= clamp.(best_policy * state, -1.0, 1.0)
    nothing
end

init_visualiser()
visualise!(model, data, controller = trained_policy_controller!)







