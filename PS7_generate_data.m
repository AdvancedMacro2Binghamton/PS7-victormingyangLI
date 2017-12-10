p.rho_1 = .5;
p.rho_2 = .3;
p.phi_1 = -.3;
p.phi_2 = 0.25;
p.beta = 5;
p.sigma_x = 1;
p.sigma_A = .25;
p.sigma_B = .7;
% "true" parameters of the model

rng(0)
T = 450;
num_W_shocks = 1;
num_V_shocks = 2;
rng(0)
W_shocks = randn(T, num_W_shocks);
V_shocks = randn(T, num_V_shocks);

num_states = 4;
num_obs = 2;
state_var = zeros(T, num_states);
observations = zeros(T, num_obs);
rand_numbers = rand(T,1);
exogenous_state = zeros(T,1);
exogenous_state(1) = 1;
cdfs = cumsum(pi,2);
for t = 1:T-1
    exogenous_state(t+1) = find(...
        rand_numbers(t) <= cdfs(exogenous_state(t),:), 1, 'first') ;
end

pol1 = exp(exogenous_process(exogenous_state)) + p.sigma_A * randn(T,1);
pol2 = p.beta*exogenous_process(exogenous_state)^2 + p.sigma_B * randn(T,1);

data = cat(3,pol1,pol2);
clearvars -except data