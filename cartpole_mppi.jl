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

function terminal_cost(data)
    return 10*running_cost(data)
end 


function running_cost(d::data)
    q = zeros(K)
    x = d.qpos[1]
    θ = d.qpos[2]
    x_dot = d.qvel[1]
    θ_dot = d.qvel[2]
    q =  10*x^2 + (500*(cos(θ)+ 1)^2) + x_dot^2 + 15*(θ_dot^2) #direct cost function that is described in the paper

    return q 
end 


global U_t = zeros(nu, T) # global controls 


init_state = vcat(data.qpos, data.qvel)

function rollout(m::model, d::data)

    for k in 1:K
        state = copy(init_state)
        qpos = state.qpos 
        qvel = state.qvel

        ϵ = randn(K, T)
        S = zeros(size(ϵ, K))  # costs
        # weights = zeros(size(ϵ), K)

        for t in 1:T 
            noise = randn(nu, T)
            d.ctrl .= U_t[:, t] + noise
            state = mjstep!(data, model)
            S .+= running_cost(data) + (λ.* (data.ctrl)' .* ϵ)  
        end
        S .+= terminal_cost(data)

        β = minimum(S)
        weights = exp.(-1 / λ * (S .- β))
        η = sum(weights) # eta is just a weight summation 

        for k in 1:K
            weights ./= η
        end 

        for t in 1:T
            U_t .+= (sum(weights)*noise for k in 1:K)
        end 

    end 
end 


function mppi_controller!(m::model, d::data)
    state = vcat(d.qpos, d.qvel)
    d.ctrl .= U_t

end 

init_visualizer()
visualise!(model, data, controller = mppi_controller!)
