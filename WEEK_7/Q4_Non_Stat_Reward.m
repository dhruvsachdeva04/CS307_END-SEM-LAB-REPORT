% Number of actions (10 arms for this bandit)
n_actions = 10;

% Epsilon-greedy parameters
epsilon = 0.1;  % Exploration rate
n_trials = 10000;  % Number of trials (iterations)

% Step-size parameter for tracking non-stationary rewards
alpha = 0.1;  % Learning rate (EWMA weight)

% Initialize reward estimates for each action
reward_estimates = zeros(1, n_actions);

% Initialize action counts (not used in update, but for tracking purposes)
action_counts = zeros(1, n_actions);

% Modified epsilon-greedy algorithm
for trial = 1:n_trials
    % Epsilon-greedy: Decide whether to explore or exploit
    if rand < epsilon
        action = randi([1, n_actions]);  % Explore: Random action
    else
        [~, action] = max(reward_estimates);  % Exploit: Choose the best estimated action
    end
    
    % Get the reward from the non-stationary bandit for the selected action
    reward = bandit_nonstat(action);
    
    % Update the action count for the selected action
    action_counts(action) = action_counts(action) + 1;
    
    % Update the reward estimate for the selected action using exponential average
    reward_estimates(action) = reward_estimates(action) + ...
        alpha * (reward - reward_estimates(action));
    
    % Optional: Display progress every 1000 steps
    if mod(trial, 1000) == 0
        disp(['Trial ', num2str(trial), ': Reward estimates are']);
        disp(reward_estimates);
    end
end

% Final display of results
disp('Final reward estimates after all trials:');
disp(reward_estimates);
disp('Number of times each action was selected:');
disp(action_counts);