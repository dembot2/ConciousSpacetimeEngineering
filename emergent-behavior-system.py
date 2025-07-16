import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from collections import defaultdict
import networkx as nx
from scipy.sparse import csr_matrix
from torch_scatter import scatter_add, scatter_mean

@dataclass
class EmergentState:
    """Represents the emergent state of the system"""
    field_tensors: torch.Tensor  # Field dynamics
    agent_states: torch.Tensor   # Individual agent states
    collective_patterns: torch.Tensor  # Emergent patterns
    interaction_graph: torch.Tensor  # Dynamic interaction network
    phase_space: torch.Tensor  # System phase space
    entropy: torch.Tensor  # System entropy measures
    metadata: Dict

class EmergentFieldDynamics(nn.Module):
    def __init__(self, field_dim: int, num_fields: int):
        super().__init__()
        self.field_dim = field_dim
        self.num_fields = num_fields
        
        # Field interaction parameters
        self.field_coupling = nn.Parameter(torch.randn(num_fields, num_fields))
        self.diffusion_rates = nn.Parameter(torch.rand(num_fields))
        
        # Non-linear field dynamics
        self.field_dynamics = nn.Sequential(
            nn.Linear(field_dim * num_fields, field_dim * num_fields),
            nn.Tanh(),
            nn.Linear(field_dim * num_fields, field_dim * num_fields)
        )
        
    def forward(self, fields: torch.Tensor, dt: float = 0.01) -> torch.Tensor:
        batch_size = fields.shape[0]
        
        # Reshape fields for processing
        fields = fields.view(batch_size, self.num_fields, -1)
        
        # Compute field interactions
        field_interactions = torch.einsum('bij,jk->bik', fields, self.field_coupling)
        
        # Apply diffusion
        laplacian = self._compute_laplacian(fields)
        diffusion = torch.einsum('bij,j->bij', laplacian, self.diffusion_rates)
        
        # Combine dynamics
        field_update = field_interactions + diffusion
        field_update = self.field_dynamics(field_update.reshape(batch_size, -1))
        field_update = field_update.view(batch_size, self.num_fields, -1)
        
        # Update fields using Euler integration
        updated_fields = fields + dt * field_update
        
        return updated_fields
    
    def _compute_laplacian(self, fields: torch.Tensor) -> torch.Tensor:
        # Compute discrete Laplacian using convolution
        kernel = torch.tensor([1., -2., 1.]).view(1, 1, -1).to(fields.device)
        laplacian = torch.nn.functional.conv1d(
            fields, kernel, padding=1, groups=self.num_fields
        )
        return laplacian

class AgentModel(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, perception_dim: int):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Perception network
        self.perception = nn.Sequential(
            nn.Linear(state_dim + perception_dim, state_dim * 2),
            nn.ReLU(),
            nn.Linear(state_dim * 2, state_dim)
        )
        
        # Decision network
        self.decision = nn.Sequential(
            nn.Linear(state_dim, action_dim),
            nn.Tanh()
        )
        
        # State update network
        self.state_update = nn.GRUCell(
            input_size=state_dim + action_dim,
            hidden_size=state_dim
        )
        
    def forward(self, state: torch.Tensor, 
                perception: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Process perception
        perceived = self.perception(torch.cat([state, perception], dim=-1))
        
        # Generate action
        action = self.decision(perceived)
        
        # Update state
        new_state = self.state_update(
            torch.cat([perceived, action], dim=-1),
            state
        )
        
        return new_state, action

class CollectiveDynamics(nn.Module):
    def __init__(self, num_agents: int, state_dim: int, 
                 field_dim: int, num_fields: int):
        super().__init__()
        self.num_agents = num_agents
        self.state_dim = state_dim
        
        # Field dynamics
        self.field_dynamics = EmergentFieldDynamics(field_dim, num_fields)
        
        # Agent models
        self.agents = nn.ModuleList([
            AgentModel(state_dim, state_dim, field_dim * num_fields)
            for _ in range(num_agents)
        ])
        
        # Interaction network
        self.interaction_net = InteractionNetwork(state_dim)
        
    def forward(self, agent_states: torch.Tensor, 
                fields: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = agent_states.shape[0]
        
        # Update fields
        new_fields = self.field_dynamics(fields)
        
        # Update agents
        new_agent_states = []
        actions = []
        
        for i, agent in enumerate(self.agents):
            # Get agent's perception of fields
            perception = self._get_perception(new_fields, agent_states[:, i])
            
            # Update agent
            new_state, action = agent(agent_states[:, i], perception)
            new_agent_states.append(new_state)
            actions.append(action)
            
        # Stack agent states and actions
        new_agent_states = torch.stack(new_agent_states, dim=1)
        actions = torch.stack(actions, dim=1)
        
        # Process interactions
        new_agent_states = self.interaction_net(new_agent_states, actions)
        
        return new_agent_states, new_fields
    
    def _get_perception(self, fields: torch.Tensor, 
                       agent_state: torch.Tensor) -> torch.Tensor:
        # Project agent state to field space
        projection = torch.einsum('bd,bfp->bfp', 
                                agent_state, 
                                fields)
        return projection.reshape(fields.shape[0], -1)

class InteractionNetwork(nn.Module):
    def __init__(self, state_dim: int):
        super().__init__()
        
        # Interaction strength predictor
        self.interaction_strength = nn.Sequential(
            nn.Linear(state_dim * 2, state_dim),
            nn.ReLU(),
            nn.Linear(state_dim, 1),
            nn.Sigmoid()
        )
        
        # State update based on interactions
        self.state_update = nn.GRUCell(
            input_size=state_dim * 2,
            hidden_size=state_dim
        )
        
    def forward(self, states: torch.Tensor, 
                actions: torch.Tensor) -> torch.Tensor:
        batch_size, num_agents, state_dim = states.shape
        
        # Compute pairwise interactions
        states_i = states.unsqueeze(2).expand(-1, -1, num_agents, -1)
        states_j = states.unsqueeze(1).expand(-1, num_agents, -1, -1)
        
        # Compute interaction strengths
        paired_states = torch.cat([states_i, states_j], dim=-1)
        interaction_weights = self.interaction_strength(
            paired_states.view(-1, state_dim * 2)
        ).view(batch_size, num_agents, num_agents)
        
        # Apply interactions
        weighted_states = torch.einsum('bijk,bij->bik', 
                                     states_j, 
                                     interaction_weights)
        
        # Update states based on interactions
        new_states = []
        for i in range(num_agents):
            new_state = self.state_update(
                torch.cat([weighted_states[:, i], actions[:, i]], dim=-1),
                states[:, i]
            )
            new_states.append(new_state)
            
        return torch.stack(new_states, dim=1)

class EmergentPatternAnalyzer:
    def __init__(self, state_dim: int):
        self.state_dim = state_dim
        
    def analyze_patterns(self, states: torch.Tensor, 
                        fields: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Compute order parameters
        order = self._compute_order_parameters(states)
        
        # Analyze phase space
        phase_space = self._analyze_phase_space(states, fields)
        
        # Compute entropy measures
        entropy = self._compute_entropy(states, fields)
        
        return {
            'order_parameters': order,
            'phase_space': phase_space,
            'entropy': entropy
        }
        
    def _compute_order_parameters(self, states: torch.Tensor) -> torch.Tensor:
        # Compute collective alignment
        mean_state = states.mean(dim=1, keepdim=True)
        alignment = torch.cosine_similarity(states, mean_state, dim=-1)
        
        # Compute clustering coefficient
        distances = torch.cdist(states, states)
        clusters = (distances < distances.mean(dim=-1, keepdim=True)).float()
        
        return torch.stack([alignment.mean(dim=-1), 
                          clusters.mean(dim=(-1, -2))])
        
    def _analyze_phase_space(self, states: torch.Tensor, 
                            fields: torch.Tensor) -> torch.Tensor:
        # Project onto principal components
        combined = torch.cat([states.view(states.shape[0], -1), 
                            fields.view(fields.shape[0], -1)], dim=-1)
        U, S, V = torch.svd(combined)
        
        return V[:, :self.state_dim]
        
    def _compute_entropy(self, states: torch.Tensor, 
                        fields: torch.Tensor) -> torch.Tensor:
        # Compute state entropy
        state_probs = torch.softmax(states, dim=-1)
        state_entropy = -(state_probs * torch.log(state_probs + 1e-10)).sum(dim=-1)
        
        # Compute field entropy
        field_probs = torch.softmax(fields.view(fields.shape[0], -1), dim=-1)
        field_entropy = -(field_probs * torch.log(field_probs + 1e-10)).sum(dim=-1)
        
        return torch.stack([state_entropy.mean(dim=-1), field_entropy])

class EmergentSimulation:
    def __init__(self, num_agents: int = 100, 
                 state_dim: int = 32, 
                 field_dim: int = 16, 
                 num_fields: int = 4):
        self.dynamics = CollectiveDynamics(
            num_agents=num_agents,
            state_dim=state_dim,
            field_dim=field_dim,
            num_fields=num_fields
        )
        
        self.analyzer = EmergentPatternAnalyzer(state_dim)
        
    def simulate(self, steps: int, 
                batch_size: int = 1) -> List[EmergentState]:
        # Initialize states
        agent_states = torch.randn(batch_size, 
                                 self.dynamics.num_agents, 
                                 self.dynamics.state_dim)
        fields = torch.randn(batch_size, 
                           self.dynamics.field_dynamics.num_fields,
                           self.dynamics.field_dynamics.field_dim)
        
        states = []
        for _ in range(steps):
            # Update dynamics
            agent_states, fields = self.dynamics(agent_states, fields)
            
            # Analyze patterns
            patterns = self.analyzer.analyze_patterns(agent_states, fields)
            
            # Create emergent state
            state = EmergentState(
                field_tensors=fields,
                agent_states=agent_states,
                collective_patterns=patterns['order_parameters'],
                interaction_graph=self._compute_interaction_graph(agent_states),
                phase_space=patterns['phase_space'],
                entropy=patterns['entropy'],
                metadata={
                    'step': len(states),
                    'num_agents': self.dynamics.num_agents
                }
            )
            
            states.append(state)
            
        return states
    
    def _compute_interaction_graph(self, 
                                 states: torch.Tensor) -> torch.Tensor:
        # Compute pairwise distances
        distances = torch.cdist(states, states)
        
        # Create adjacency matrix based on proximity
        threshold = distances.mean(dim=-1, keepdim=True)
        adjacency = (distances < threshold).float()
        
        return adjacency

def visualize_emergence(states: List[EmergentState], 
                       save_path: Optional[str] = None):
    import matplotlib.pyplot as plt
    
    # Plot order parameters over time
    order_params = torch.stack([s.collective_patterns for s in states])
    plt.figure(figsize=(10, 6))
    plt.plot(order_params[:, 0].cpu().numpy(), 
             label='Alignment')
    plt.plot(order_params[:, 1].cpu().numpy(), 
             label='Clustering')
    plt.xlabel('Time Step')
    plt.ylabel('Order Parameter')
    plt.legend()
    
    if save_path:
        plt.savefig(f'{save_path}_order.png')
    plt.close()
    
    # Plot entropy evolution
    entropy = torch.stack([s.entropy for s in states])
    plt.figure(figsize=(10, 6))
    plt.plot(entropy[:, 0].cpu().numpy(), 
             label='State Entropy')
    plt.plot(entropy[:, 1].cpu().numpy(), 
             label='Field Entropy')
    plt.xlabel('Time Step')
    plt.ylabel('Entropy')
    plt.legend()
    
    if save_path:
        plt.savefig(f'{save_path}_entropy.png')
    plt.close()
