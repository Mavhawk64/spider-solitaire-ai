import AiTrainer
import time

class MultiAgentTrainer:
    def __init__(self, num_agents=10):
        # list of agents with different seeds
        # seed = the clock time like in random library
        self.agents = [AiTrainer.AiTrainer(seed = time.time_ns()) for _ in range(num_agents)]

    def train(self, num_games_per_agent):
        for _ in range(num_games_per_agent):
            for agent in self.agents:
                agent.train(1)

# Example usage
multi_agent_trainer = MultiAgentTrainer(num_agents=10)
multi_agent_trainer.train(num_games_per_agent=100)
