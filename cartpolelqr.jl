using MuJoCo
using LinearAlgebra
using MatrixEquations


model = load_model("cartpole.xml")
data = init_data(model)


function lqr(m::Model, d::Data)
    # number of states and control inputs
    nx = 2*model.nq
    nu = model.nu
    # finite-difference parameters
    c = 1e-6
    centred = true
    # compute the matrices (A, B)
    A = mj_zeros(nx,nx)
    B = mj_zeros(nx,nu)
    mjd_transitionFD(model, data, c, centred, A, B, nothing, nothing)

    Q = diagm([1.0, 10.0, 1.0, 5.0]) # Weights for the state vector
    R = diagm([1.0]) 


    S = zeros(nx, nu) # cross term penalty costs
    _, _, K, _ = ared(A,B,R,Q,S)
    return K 
end 


function lqr_balance!(m::Model, d::Data)
    state = vcat(d.qpos, d.qvel)
    K = lqr(m,d)
    d.ctrl .= -K * state
    nothing
end

H = 50 # H is the horizon 
function MPC_lqr!(m::Model, d::Data)
    nx = 2 * model.nq 
    nu = model.nu 

    #predicted 
    pred_states = zeros(nx, H+1)
    U_pred = zeros(nu, H) # initial controls are just zeros

    x0 = vcat(d.qpos, d.qvel)
    pred_states[:,1] = x0 
    
    m_sim = load_model("cartpole.xml")
    d_sim = init_data(m_sim)

    # copy in the state from the actual sim
    d_sim.qpos .= d.qpos
    d_sim.qvel .= d.qvel

    # run a prediction scheme 
    for t in 1:H 
        
        #lQR problem for sim 
        K = lqr(m_sim, d_sim)
        U_pred[:, t] = -K * pred_states[:, t]
        
        U_pred[:, t] = clamp.(U_pred[:, t], -1.0, 1.0) # constraints based on the xml file 

        d_sim.ctrl .= U_pred[:, t]
        step!(m_sim, d_sim)

        pred_states[:, t+1] = vcat(d_sim.qpos, d_sim.qvel)
    end 

    d.ctrl .= U_pred[:, 1]
    nothing 
end 

init_visualiser()
visualise!(model, data, controller = MPC_lqr!)



