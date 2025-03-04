using MuJoCo 
using Flux 
using LinearAlgebra 

model = load_model("cartpole.xml")
data = init_data(model)
num_observations = model.qpos + model.qvel # number of observable states 
num_actions = model.nu # number of actuators 

base_policy = zeros(num_observations, num_actions)
best_reward = -Inf
best_policy = base_policy

num_trajectories = 10 
num_episodes = 100 # total training episodes 
max_steps = 1000 # maximum steps per trajectory 
noise_scale = 0.05 # for policy updates
learning_rate = 0.01

# outermost is training loop -> policy parameters change per steps
# outer for loop -> sampling policies 
# inner for loop ->  using policies to get trajectories

# Training loop is outermost where the policy params change through steps. 
# Outer loop is sampling policies. Inter loop is using policy to get trajectories

for episode in 1:num_episodes
    policies = []
    rewards = [] 

    for i in 1:num_trajectories
        # get one or more trajectories and their reward -> add noise every time to the policy 
        mj_resetData(model, data)

        policy = base_policy .+ randn(size(base_policy)) * noise_scale
        push!(policies, policy)

        total_reward = 0.0

        for step in 1:max_steps 
            # get reward/ observations (multiply with policy to get actions -> apply actions to sample_model_and_data)
            # step forward in time (get trajectory/ get reward)
            observations = vcat(model.qpos, model.qvel) 
            action = policy * observation #simple linear policy

            data.ctrl .= clamp(action, -1.0, 1.0)
        
            mj_step(model, data)

            angle_reward = cos(data.qpos[2]) # cos(pole angle)
            pos_penalty = 0.1 * abs(data.qpos[1]) #keep the cart centered
            reward = angle_reward - pos_penalty
            total_reward += reward 

            if reward >= best_reward
                best_reward = reward
                best_policy = copy(policy)
            end

        end 
        push!(rewards, total_reward)
    end 
    

    #combine -> weight all the policies with the rewards generated to approximate a gradient direction and take a step 
    # training loop 
    rewards = (rewards .- mean(rewards)) / (std(rewards) + 1e-8)

    gradient = zeros(size(base_policy))
    for i in 1:num_trajectories
    noise = policies[i] - base_policy 
    gradient += noise*rewards[i]
    end 
    base_policy += learning_rate * gradient/num_trajectories 

    if episode % 10 == 0
        println("Episode $episode, Avg Reward: $(mean(rewards)), Best: $best_reward")
    end

end 


function trained_policy_controller!(m::Model, d::Data)
    state = vcat(d.qpos, d.qvel)
    d.ctrl .= clamp.(best_policy * state, -1.0, 1.0)
    nothing
end

init_visualiser()
visualise!(model, data, controller = trained_policy_controller!)







