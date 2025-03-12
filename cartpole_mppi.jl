using MuJoCo 


model = load_model("cartpole.xml")
data = init_data(model)

global const K = 10  #number of trajectories 
global const T = 100 #number of time steps 

# cost function parameters 
const Σ = 0
const ϕ = 0
const q = 0  
const λ = 0 

nx = 2 * model.nx # number of states
nu = model.nu # number of controls 

global U_t = zeros(nu, T-1) # global controls 


init_state = vcat(data.qpos, data.qvel)
S = zeros(size(ϵ, K))


function rollout(m::model, d::data)

    for k in 0:K-1 
        state = copy(init_state)
        
        ϵ = randn(K, T-1)

        for t in 1:T 
            d.ctrl .= U_t[:, t] + noise
            state = mjstep!(data, model)
            S .+= running_cost(data) + (λ.* (data.ctrl)' .* ϵ)  
        end
        S .+= terminal_cost(data)
    end 

end 

β = minimum(S)
η = 

for k in 0:K-1
    weights = zeros(size(ϵ), K)
    weights .+= 1/η .* exp(-1 / λ (S .- β))
end 

for t in 1:T-1
    U_t .+= weights .* ϵ
end 


function terminal_cost(data)

end 



function running_cost(data)
    q = zeros(K)
    x = data.qpos[1]
    θ = data.qpos[2]
    x_dot = data.qvel[1]
    θ_dot = data.qvel[2]
    q =  10*x^2 + (500*(cos(θ)+ 1)^2) + x_dot^2 + 15*(θ_dot^2)
    return q 
end 

