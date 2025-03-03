using MuJoCo
model = load_model("cartpole.xml")
data = init_data(model)
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

using LinearAlgebra
Q = diagm([1.0, 10.0, 1.0, 5.0]) # Weights for the state vector
R = diagm([1.0]) 

using MatrixEquations
S = zeros(nx, nu) # cross term penalty costs
_, _, K, _ = ared(A,B,R,Q,S)

function lqr_balance!(m::Model, d::Data)
    state = vcat(d.qpos, d.qvel)
    d.ctrl .= -K * state
    nothing
end

init_visualiser()
visualise!(model, data, controller = lqr_balance!)



