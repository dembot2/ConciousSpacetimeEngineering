import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import math

@dataclass
class EmotionalGeometryState:
    """Represents the emotional-geometric state of the system"""
    emotion_vectors: torch.Tensor  # Emotional state vectors
    geometric_forms: torch.Tensor  # Generated geometric patterns
    resonance_field: torch.Tensor  # Emotional resonance field
    harmonic_metrics: torch.Tensor  # Measures of geometric harmony
    flow_patterns: torch.Tensor    # Dynamic flow patterns
    metadata: Dict

class EmotionalGeometryGenerator(nn.Module):
    def __init__(self, 
                 emotion_dim: int = 8,
                 geometry_dim: int = 16,
                 num_harmonics: int = 4):
        super().__init__()
        self.emotion_dim = emotion_dim
        self.geometry_dim = geometry_dim
        self.num_harmonics = num_harmonics
        
        # Emotional field generator
        self.emotion_field = nn.Sequential(
            nn.Linear(emotion_dim, geometry_dim * 2),
            nn.SiLU(),  # Smoother activation for emotional flow
            nn.Linear(geometry_dim * 2, geometry_dim * num_harmonics),
            nn.Tanh()
        )
        
        # Geometric pattern synthesizer
        self.geometry_synth = nn.Sequential(
            nn.Linear(geometry_dim * num_harmonics, geometry_dim * 4),
            nn.LayerNorm(geometry_dim * 4),
            nn.SiLU(),
            nn.Linear(geometry_dim * 4, geometry_dim * 2)
        )
        
        # Harmonic resonance network
        self.harmonic_net = nn.ModuleList([
            nn.Linear(geometry_dim * 2, geometry_dim)
            for _ in range(num_harmonics)
        ])
        
        # Flow pattern generator
        self.flow_generator = nn.GRU(
            input_size=geometry_dim * 2,
            hidden_size=geometry_dim,
            num_layers=2,
            batch_first=True
        )
        
    def forward(self, emotional_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = emotional_state.shape[0]
        
        # Generate emotional field
        emotional_field = self.emotion_field(emotional_state)
        emotional_field = emotional_field.view(batch_size, self.num_harmonics, -1)
        
        # Synthesize geometric patterns
        geometry_base = self.geometry_synth(emotional_field.reshape(batch_size, -1))
        
        # Generate harmonic resonances
        harmonics = []
        for harmonic_layer in self.harmonic_net:
            harmonic = harmonic_layer(geometry_base)
            harmonics.append(harmonic)
        
        harmonic_field = torch.stack(harmonics, dim=1)
        
        # Generate flow patterns
        flow_sequence = geometry_base.unsqueeze(1).expand(-1, self.num_harmonics, -1)
        flow_patterns, _ = self.flow_generator(flow_sequence)
        
        return harmonic_field, flow_patterns

class EmotionalGeometrySystem:
    def __init__(self, 
                 emotion_dim: int = 8,
                 geometry_dim: int = 16,
                 num_harmonics: int = 4):
        self.generator = EmotionalGeometryGenerator(
            emotion_dim=emotion_dim,
            geometry_dim=geometry_dim,
            num_harmonics=num_harmonics
        )
    
    def generate_patterns(self, 
                         emotional_input: torch.Tensor,
                         steps: int = 16) -> List[EmotionalGeometryState]:
        states = []
        batch_size = emotional_input.shape[0]
        
        # Initialize emotional state
        current_emotion = emotional_input
        
        for _ in range(steps):
            # Generate geometric patterns
            harmonic_field, flow_patterns = self.generator(current_emotion)
            
            # Compute resonance field
            resonance = self._compute_resonance(harmonic_field, flow_patterns)
            
            # Calculate harmonic metrics
            harmony = self._analyze_harmony(harmonic_field, resonance)
            
            # Create state
            state = EmotionalGeometryState(
                emotion_vectors=current_emotion,
                geometric_forms=harmonic_field,
                resonance_field=resonance,
                harmonic_metrics=harmony,
                flow_patterns=flow_patterns,
                metadata={
                    'step': len(states),
                    'num_harmonics': self.generator.num_harmonics
                }
            )
            
            states.append(state)
            
            # Evolve emotional state
            current_emotion = self._evolve_emotion(current_emotion, resonance)
            
        return states
    
    def _compute_resonance(self,
                          harmonic_field: torch.Tensor,
                          flow_patterns: torch.Tensor) -> torch.Tensor:
        # Combine harmonic field and flow patterns to create resonance
        harmonic_energy = torch.sum(harmonic_field ** 2, dim=-1, keepdim=True)
        flow_energy = torch.sum(flow_patterns ** 2, dim=-1, keepdim=True)
        
        resonance = torch.sigmoid(harmonic_energy) * torch.tanh(flow_energy)
        return resonance
    
    def _analyze_harmony(self,
                        harmonic_field: torch.Tensor,
                        resonance: torch.Tensor) -> torch.Tensor:
        # Compute various harmony metrics
        field_coherence = torch.mean(torch.std(harmonic_field, dim=1), dim=1)
        resonance_stability = torch.mean(resonance, dim=1)
        cross_harmony = torch.mean(torch.cross(
            harmonic_field[:, :-1],
            harmonic_field[:, 1:],
            dim=2
        ), dim=(1, 2))
        
        return torch.stack([field_coherence,
                          resonance_stability,
                          cross_harmony], dim=1)
    
    def _evolve_emotion(self,
                       emotion: torch.Tensor,
                       resonance: torch.Tensor) -> torch.Tensor:
        # Evolve emotional state based on resonance
        resonance_influence = torch.mean(resonance, dim=1)
        evolved_emotion = emotion + 0.1 * resonance_influence
        return torch.tanh(evolved_emotion)  # Keep emotions bounded

def render_emotional_geometry(states: List[EmotionalGeometryState],
                            save_path: Optional[str] = None):
    """Visualize emotional geometry patterns"""
    import matplotlib.pyplot as plt
    
    # Plot harmonic metrics evolution
    harmony = torch.stack([s.harmonic_metrics for s in states])
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(harmony[:, 0].cpu().numpy(), label='Field Coherence')
    plt.plot(harmony[:, 1].cpu().numpy(), label='Resonance Stability')
    plt.plot(harmony[:, 2].cpu().numpy(), label='Cross Harmony')
    plt.xlabel('Time Step')
    plt.ylabel('Harmonic Metrics')
    plt.legend()
    
    # Plot resonance field patterns
    plt.subplot(1, 2, 2)
    resonance = states[-1].resonance_field[0].cpu().numpy()
    plt.imshow(resonance, cmap='viridis', aspect='auto')
    plt.colorbar(label='Resonance Strength')
    plt.xlabel('Pattern Dimension')
    plt.ylabel('Harmonic Layer')
    plt.title('Final Resonance Field')
    
    if save_path:
        plt.savefig(f'{save_path}_emotional_geometry.png')
    plt.close()

# Example usage:
if __name__ == "__main__":
    # Create emotional geometry system
    system = EmotionalGeometrySystem(
        emotion_dim=8,
        geometry_dim=16,
        num_harmonics=4
    )
    
    # Generate example emotional input
    emotional_input = torch.randn(1, 8)  # Batch size 1, 8 emotional dimensions
    
    # Generate patterns
    states = system.generate_patterns(emotional_input, steps=32)
    
    # Visualize results
    render_emotional_geometry(states, save_path="emotional_geometry")
