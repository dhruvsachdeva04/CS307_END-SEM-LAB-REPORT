% Explanation:
% Persistent Variables in this code :-
% mean_rewards: Stores the current mean rewards for the 10 actions. These are initialized to zero and updated with each function call.
% step_count: Tracks the number of time steps (or function calls).
% Random Walk Update:
At each call, a small random value with mean 0 and standard deviation 0.01 is added to each action''s mean reward.
% Reward Generation:
% When an action is selected, the function returns a reward that is normally distributed with a mean equal to the current mean reward of the action and a standard deviation of 1.

% CODE

function [value] = bandit_nonstat(action)
    % Persistent variables to store mean rewards and step count across function calls
    persistent mean_rewards;
    persistent step_count;
    
    % Initialize variables on the first function call
    if isempty(mean_rewards)
        mean_rewards = zeros(1, 10);  % Start with mean rewards of 0 for all actions
        step_count = 0;
    end
    
    % Standard deviation for random walk
    stddev_random_walk = 0.01;
    
    % Update step count
    step_count = step_count + 1;
    
    % Update each action''s mean reward by adding a small random increment
    mean_rewards = mean_rewards + normrnd(0, stddev_random_walk, [1, 10]);
    
    % The selected action''s reward is drawn from a normal distribution centered on its current mean reward
    value = normrnd(mean_rewards(action), 1);
    
    % Display current step and updated mean rewards (optional, for debugging)
    if mod(step_count, 100) == 0
        disp(['Step ', num2str(step_count), ': Mean rewards are']);
        disp(mean_rewards);
    end
end