import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class DeepQNetwork(nn.Module):
    """
    A fully connected Deep Q-Network (DQN) designed to process symbolic game states
    from a Spider Solitaire environment and output Q-values for a set of discrete actions.

    Parameters:
    -----------
    lr : float
        The learning rate for the Adam optimizer.

    input_dims : int
        The size of the flattened game state vector. Default is 805, based on:
            - Tableau: 10 columns x 20 cards each x 4 features = 800
                (Each card is encoded as: [rank, suit, face_up, known])
            - Draw pile size (normalized): 1
            - Completed sets (normalized): 1
            - Can draw / can undo (booleans): 2
            - Number of empty piles (normalized): 1

    fc1_dims : int
        The number of neurons in the first fully connected layer. Default is 512.

    fc2_dims : int
        The number of neurons in the second fully connected layer. Default is 512.

    n_actions : int
        The number of discrete actions the agent can take. This should be the total
        number of distinct move choices the agent may select from (e.g., draw, undo,
        or all valid (source, target, bundle) move combinations).
        Default is 1172:
            = 10 source piles
            * (10-1) target piles
            * 13 bundle_sizes
            + 2 (draw, undo)

    Attributes:
    -----------
    fc1, fc2, fc3 : nn.Linear
        Fully connected layers.
    optimizer : T.optim.Optimizer
        The optimizer used for training.
    loss : nn.Module
        The loss function (Mean Squared Error for Q-learning).
    device : torch.device
        GPU if available, else CPU.
    """

    def __init__(self, lr, input_dims=805, fc1_dims=512, fc2_dims=512, n_actions=1172):
        super(DeepQNetwork, self).__init__()

        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.fc3 = nn.Linear(fc2_dims, n_actions)

        self.optimizer = T.optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state):
        """
        Perform a forward pass through the network.

        Parameters:
        -----------
        state : torch.Tensor
            The input game state tensor (shape: [batch_size, input_dims]).

        Returns:
        --------
        torch.Tensor
            A tensor of Q-values for each possible action (shape: [batch_size, n_actions]).
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        actions = self.fc3(x)
        return actions


class Agent(object):
    def __init__(
        self,
        gamma,
        epsilon,
        alpha,
        maxMemorySize,
        epsEnd=0.05,
        replace=10000,
        actionSpace=None,
        input_dims=805,
        fc1_dims=512,
        fc2_dims=512,
        n_actions=1172,
    ):
        self.GAMMA = gamma
        self.EPSILON = epsilon
        self.EPS_END = epsEnd
        self.actionSpace = actionSpace
        self.memSize = maxMemorySize
        self.steps = 0
        self.learn_step_counter = 0
        self.memory = []
        self.memCntr = 0
        self.replace_target_cnt = replace
        # fmt: off
        self.Q_eval = DeepQNetwork(alpha, input_dims, fc1_dims, fc2_dims, n_actions)
        self.Q_next = DeepQNetwork(alpha, input_dims, fc1_dims, fc2_dims, n_actions)
        # fmt: on

    def storeTransition(self, state, action, reward, state_):
        if self.memCntr < self.memSize:
            self.memory.append([state, action, reward, state_])
        else:
            self.memory[self.memCntr % self.memSize] = [state, action, reward, state_]
        self.memCntr += 1

    def chooseAction(self, observation, valid_action_indices):
        """
        Chooses an action using ε-greedy strategy, only among valid actions.

        Parameters:
        -----------
        observation : torch.Tensor
            A tensor representing the current game state. Should have shape [1, input_dims].

        valid_action_indices : List[int]
            List of valid action indices based on current game state.

        Returns:
        --------
        int : The index of the selected action (0 to n_actions - 1)
        """
        self.steps += 1
        rand = np.random.random()

        # Get all Q-values from the policy network
        q_vals = self.Q_eval.forward(observation).squeeze()  # shape: [n_actions]

        if rand < 1 - self.EPSILON:
            # Mask all invalid actions to -inf so they’re never chosen
            masked_q_vals = T.full_like(q_vals, float("-inf"))
            masked_q_vals[valid_action_indices] = q_vals[valid_action_indices]
            action_idx = T.argmax(masked_q_vals).item()
        else:
            # Randomly select one of the valid actions
            action_idx = np.random.choice(valid_action_indices)

        return action_idx

    def learn(self, batch_size=32):
        if self.memCntr < batch_size:
            return  # not enough data

        if (
            self.replace_target_cnt is not None
            and self.learn_step_counter % self.replace_target_cnt == 0
        ):
            self.Q_next.load_state_dict(self.Q_eval.state_dict())

        self.Q_eval.optimizer.zero_grad()

        mem_range = min(self.memCntr, self.memSize)
        batch_indices = np.random.choice(mem_range, batch_size, replace=False)
        miniBatch = [self.memory[i] for i in batch_indices]

        states, actions, rewards, next_states = zip(*miniBatch)

        states = T.tensor(states, dtype=T.float32).to(self.Q_eval.device)
        actions = T.tensor(actions, dtype=T.int64).to(self.Q_eval.device)
        rewards = T.tensor(rewards, dtype=T.float32).to(self.Q_eval.device)
        next_states = T.tensor(next_states, dtype=T.float32).to(self.Q_next.device)

        q_pred = self.Q_eval(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        q_next = self.Q_next(next_states).max(dim=1)[0]
        q_target = rewards + self.GAMMA * q_next

        loss = self.Q_eval.loss(q_pred, q_target)
        loss.backward()
        self.Q_eval.optimizer.step()
        self.learn_step_counter += 1

        # Epsilon decay
        if self.steps > 500:
            self.EPSILON = max(self.EPS_END, self.EPSILON - 1e-4)
