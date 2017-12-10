%%%% Extremely simple sample application of a Metropolis hastings algorithm
%%%% (this script) and a particle filter to approximate the likelihood
%%%% function empirically ('PS7_model_llh.m').
%%%% Instructions: Load 'PS7_data.mat'. Specify priors, step sizes and number of
%%%% particles below, and run this script. Major credit goes to Prof. Kuhn.
clear, clc, close all
tic
% likelihood simulation parameters:
N = 1000; % number of particles
T = 400; % length of time series (given by data)

% data needs to be provided:
% data = cell2mat(struct2cell(load('PS7_data.mat')));
load data

% priors:
prior.rho_1 = @(x) normpdf(x, 0, 3);
prior.rho_2 = @(x) lognpdf(x, 0, 3);
prior.phi_1 = @(x) lognpdf(x, 0, 3);
prior.phi_2 = @(x) lognpdf(x, 0, 3);
prior.beta = @(x) 1;
prior.sigma_x = @(x) lognpdf(x, -1/2, 2);
prior.sigma_A = @(x) lognpdf(x, -1/2, 2);
prior.sigma_B = @(x) lognpdf(x, -1/2, 2);
prior.all = @(p) log(prior.rho_1(p(1))) + log(prior.rho_2(p(2))) + ...
    log(prior.phi_1(p(3))) + log(prior.phi_2(p(4))) + ...
    log(prior.beta(p(5))) + log(prior.sigma_x(p(6))) + ...
    log(prior.sigma_A(p(7))) + log(prior.sigma_B(p(8)));

% proposals according to random walk with parameter sd's:
prop_sig.rho_1 = 1;
prop_sig.rho_2 = 1;
prop_sig.phi_1 = 1;
prop_sig.phi_2 = 1;
prop_sig.beta = 1;
prop_sig.sigma_x = 1;
prop_sig.sigma_A = 1;
prop_sig.sigma_B = 1;
prop_sig.all = .04 * [prop_sig.rho_1 prop_sig.rho_2 prop_sig.phi_1 ...
    prop_sig.phi_2 prop_sig.sigma_x prop_sig.beta prop_sig.sigma_A ...
    prop_sig.sigma_B];

% initial values for parameters
init_params = [.5 .5 .5 .5 4 1 1 1];

% length of sample
M = 5000;
% M_burnin = 1000;
acc_rate = zeros(M, 8);

llhs = zeros(M, 1);
parameters = zeros(M, 8);
parameters(1, :) = init_params;

% evaluate model with initial parameters
log_prior = prior.all(parameters(1, :));
llh = PS7_model_llh(parameters(1, :), data, N, T);
llhs(1) = log_prior + llh;

% sample:
rng(1)
oneatatime = 0;
proposal_chance = log(rand(M, 1));
prop_step = randn(M, 8);
for m = 2:M
    % proposal draw:
    prop_param = parameters(m-1, :);
    vary_param = mod(m, 8) + 1;
    if oneatatime
        prop_param(vary_param) = prop_param(vary_param) + ...
            prop_step(m, vary_param) .* prop_sig.all(vary_param);
    else
        prop_param = prop_param + prop_step(m, :) .* prop_sig.all(vary_param);
    end
    
    % evaluate prior and model with proposal parameters:
    prop_prior = prior.all(prop_param);
    if prop_prior > -Inf % theoretically admissible proposal
        prop_llh = PS7_model_llh(prop_param, data, N, T);
        llhs(m) = prop_prior + prop_llh;
        if llhs(m) - llhs(m-1) > proposal_chance(m)
            accept = 1;
        else
            accept = 0;
        end
    else % reject proposal since disallowed by prior
        accept = 0;
    end
    
    % update parameters (or not)
    if accept
        parameters(m, :) = prop_param;
        acc_rate(m, :) = 1;
    else
        parameters(m, :) = parameters(m-1, :);
        llhs(m) = llhs(m-1);
    end
    
    waitbar(m / M)
end

toc

acc = sum(acc_rate) / M
str={'\rho_1', '\rho_2', '\phi_1', '\phi_2', ...
     '\beta', '\sigma_x', '\sigma_A','\sigma_B'};
for i=1:8
    subplot(2, 4, i)
    hist(parameters(:, i), 50);
    title(str{i});
end