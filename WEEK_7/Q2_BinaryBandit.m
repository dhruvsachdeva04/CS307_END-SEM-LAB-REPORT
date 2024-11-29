% IMP POINTS
% Random Selection of Action: MATLAB uses randi([1, 2]) to select random actions between 1 and 2.
% Reward Retrieval: The functions binaryBanditA(action) and binaryBanditB(action) are called to get rewards for the chosen actions.
% Max Function: max() function returns both the maximum value and its index. We use ~, action_A and ~, action_B to get the action with the highest estimated reward.
% You can run this MATLAB script, ensuring that binaryBanditA.m and binaryBanditB.m are in the same directory, and the code will apply the epsilon-greedy algorithm to maximize the expected reward.

% Number of actions (2 actions for each bandit)
n_actions = 2;

% Epsilon-greedy parameters
epsilon = 0.1;  % Exploration rate
n_trials = 1000;  % Number of trials

% Initial estimates for both binaryBanditA and binaryBanditB
reward_estimates_A = zeros(1, n_actions);
reward_estimates_B = zeros(1, n_actions);
action_counts_A = zeros(1, n_actions);
action_counts_B = zeros(1, n_actions);

% Epsilon-greedy algorithm
for trial = 1:n_trials
    % Decide whether to explore or exploit for each bandit
    if rand < epsilon
        action_A = randi([1, n_actions]);  % Explore: Random action for bandit A
        action_B = randi([1, n_actions]);  % Explore: Random action for bandit B
    else
        [~, action_A] = max(reward_estimates_A);  % Exploit: Choose best estimated action for bandit A
        [~, action_B] = max(reward_estimates_B);  % Exploit: Choose best estimated action for bandit B
    end
    
    % Get the reward from both bandits
    reward_A = binaryBanditA(action_A);  % Call the binaryBanditA function with the selected action
    reward_B = binaryBanditB(action_B);  % Call the binaryBanditB function with the selected action
    
    % Update estimates for Bandit A
    action_counts_A(action_A) = action_counts_A(action_A) + 1;
    reward_estimates_A(action_A) = reward_estimates_A(action_A) + ...
        (reward_A - reward_estimates_A(action_A)) / action_counts_A(action_A);
    
    % Update estimates for Bandit B
    action_counts_B(action_B) = action_counts_B(action_B) + 1;
    reward_estimates_B(action_B) = reward_estimates_B(action_B) + ...
        (reward_B - reward_estimates_B(action_B)) / action_counts_B(action_B);
end

% Output final estimates
disp('Final reward estimates for Binary Bandit A:');
disp(reward_estimates_A);

disp('Final reward estimates for Binary Bandit B:');
disp(reward_estimates_B);