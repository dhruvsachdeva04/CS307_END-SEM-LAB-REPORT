% Explanation
% reward_estimates: Keeps track of the current estimated reward for each of the 10 actions. Initially, all estimates are set to 0.
% action_counts: Counts how many times each action has been selected.
% Epsilon-Greedy Decision : With probability 𝜖 = 1, we randomly select an action (exploration), and with probability 1 - ϵ, we select the action with the highest estimated reward (exploitation). The max() function returns both the maximum estimated reward and the corresponding action index.
% Reward Retrieval and Update : the reward for the selected action is obtained by calling bandit_nonstat(action), which also internally updates the mean rewards based on the random walk. The reward estimate for the action is updated using the incremental average formula :- 
% New estimate = [Old estimate + ((Reward - Old estimate) / action Count) ]
% The code prints the reward estimates every 100 trials for tracking progress. At the end of all trials, it displays the final reward estimates and the number of times each action was selected.

% CODE

% Number of actions (10 arms for this bandit)
n_actions = 10;

% Epsilon-greedy parameters
epsilon = 0.1;  % Exploration rate
n_trials = 1000;  % Number of trials (iterations)

% Initialize reward estimates and action counts for each action
reward_estimates = zeros(1, n_actions);
action_counts = zeros(1, n_actions);

% Epsilon-greedy algorithm
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
    
    % Update the reward estimate for the selected action using incremental average
    reward_estimates(action) = reward_estimates(action) + ...
        (reward - reward_estimates(action)) / action_counts(action);
    
    % Optional: Display progress every 100 steps
    if mod(trial, 100) == 0
        disp(['Trial ', num2str(trial), ': Reward estimates are']);
        disp(reward_estimates);
    end
end

% Final display of results
disp('Final reward estimates after all trials:');
disp(reward_estimates);
disp('Number of times each action was selected:');
disp(action_counts);