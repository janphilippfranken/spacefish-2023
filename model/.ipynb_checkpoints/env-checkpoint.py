import numpy as np


class Env():
    
    def __init__(self, n_trials: int, n_agents: int, structure: np.array):
        """Update environment based on agent observation and communication.
          
          Args:
          
          Returns:
        """
        self.agents = n_agents
        self.trials = n_trials
        self.observations =  np.zeros((self.trials, self.agents))
        self.communications = np.zeros((self.trials, self.agents))
        self.structure = structure
        
    def update_env(self, trial: int, agent_id: int, observation: int, communication: int):
        """Update environment based on agent observation and communication.
          
          Args:
          
          Returns:
        """
        self.observations[trial, agent_id] = observation
        self.communications[trial, agent_id] = communication
        
    def get_env(self, trial: int, agent_id: int):
        """Get available observations and communications from environment for agent.
          
          Args:
          
          Returns:
        """
        observation = self.observations[:trial + 1]
        communication = self.communications[:trial, np.where(self.structure[:,agent_id] == 1)[0]]
        parents = {}
        for agent in range(self.communications.shape[1]):
            parents[agent] =  np.where(self.structure[:,agent] == 1)[0]
        return observation, communication, parents