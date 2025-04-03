using MuJoCo 
using Statistics 
using LinearAlgebra 
using UnicodePlots 

model = load_model("hopper.xml")
data = init_data(model)


function stand_reward(data)
    reward = 0.0 
    
    upright_bonus = 1.0 
    height = data.qpos[2]
    t_height = 0.0 
    if height > t_height
        reward += upright_bonus
        reward += abs(data.qvel[1]-2)  #data.qvel[1]*4
    else
        reward -= abs(t_height-height)
    end
    # reward -= 1e-3 * norm(data.ctrl)^2  # standing tends to get substantially better when the control cost is taken away 

    return reward 
end 

function mpc_controller!(model, data)
    state = vcat(data.qpos, data.qvel)
    
    H = 50 # Horizon 
    num_candidates = 100 
    action_sequences = [] # Renamed for clarity

    # Generate random action sequences
    for _ in 1:num_candidates
        sequence = randn(model.nu, H)
        sequence = clamp.(sequence, -1.0, 1.0) # Clamp the sequence
        push!(action_sequences, sequence) # Push the full sequence
    end 
     
    best_reward = -Inf 
    best_actions = nothing 

    sim_data = init_data(model)
    
    # Evaluate each candidate sequence
    for actions in action_sequences 
        sim_data.qpos .= data.qpos 
        sim_data.qvel .= data.qvel

        total_reward = 0.0 

        for t in 1:H 
           sim_data.ctrl .= actions[:, t] 
           step!(model, sim_data)

           reward = stand_reward(sim_data)
           total_reward += reward
        end 

        if total_reward > best_reward
            best_reward = total_reward
            best_actions = actions
        end 
    end 

    if best_actions !== nothing
        data.ctrl .= best_actions[:, 1]
    else
        data.ctrl .= zeros(model.nu)
    end
    
    nothing
end


mj_resetData(model, data)
init_visualiser()
visualise!(model, data, controller = mpc_controller!)