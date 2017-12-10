% Auxiliary function to 'PS7_McMc_sampler.m'. Credit goes to Prof. Kuhn.
function [LLH] = PS7_model_llh(params, data, N, T)
p.rho_1 = params(1);
p.rho_2 = params(2);
p.phi_1 = params(3);
p.phi_2 = params(4);
p.beta = params(5);
p.sigma_x = params(6);
p.sigma_A = params(7);
p.sigma_B = params(8);

T = min(T, length(data));
data_logA = log(data(:, 1));
data_B = data(:, 2);

% What's the long run distribution over state (X(t), X(t-1), X(t-2))?
rng(0)
lr_sim = 5000;
x_distn = zeros(lr_sim + 3, 1);
distn_shocks = p.sigma_x + randn(lr_sim + 3, 1);
for t = 3:lr_sim + 3
    x_distn(t) = p.rho_1 * x_distn(t-1) + p.rho_2 * x_distn(t-2) + ...
        p.phi_1 * distn_shocks(t-1) + p.phi_2 * distn_shocks(t-2);
end

particles = zeros(T, N, 6);
llhs = zeros(T, 1);
init_sample = randsample(lr_sim, N);
particles(1, :, 1) = x_distn(init_sample + 2);
particles(1, :, 2) = x_distn(init_sample + 1);
particles(1, :, 3) = x_distn(init_sample);
particles(1, :, 4) = distn_shocks(init_sample + 2);
particles(1, :, 5) = distn_shocks(init_sample + 1);
particles(1, :, 6) = distn_shocks(init_sample);
likelihoods = normpdf(data_logA(1), particles(1, :, 1), p.sigma_A) .* ...
    normpdf(data_B(1), p.beta * particles(1, :, 1) .^ 2, p.sigma_B);
llhs(1) = log(sum(likelihoods)) - log(N);
    
% predict, filter, update particles and collect the likelihood 
for t = 2:T
    %%% Prediction:
    shocks = p.sigma_x * randn(1,N);
    particles(t, :, 1) = p.rho_1 * particles(t-1, :, 1) + p.rho_2 * ...
        particles(t-1, : , 2) + p.phi_1 * particles(t-1, :, 4) + p.phi_2...
        * particles(t-1, :, 5) + shocks;
    
    %%% Filtering:
    likelihoods = normpdf(data_logA(t), particles(t, :, 1), p.sigma_A) .* ...
        normpdf(data_B(t), p.beta * particles(t, :, 1) .^ 2, p.sigma_B);
    sampling_weights = exp(log(likelihoods) - log(sum(likelihoods)));
    if sum(likelihoods) == 0
        sampling_weights(:) = 1 / length(sampling_weights);
    end
    % store the log(mean likelihood)
    llhs(t) = log(sum(likelihoods)) - log(N);
    
    %%% Sampling:
    samples = randsample(N, N, true, sampling_weights);
    particles(t, :, :) = particles(t, samples, :);
    
end
LLH = sum(llhs);