"""
AlphaZero-style neural network for chess.

This module implements a deep residual neural network architecture similar
to AlphaZero, with a shared trunk and two heads:
- Policy head: Outputs move probabilities
- Value head: Outputs position evaluation

Architecture:
    Input (8x8x119 planes) -> Conv Block -> N x Residual Blocks -> Policy/Value Heads

The network uses:
- Batch normalization for stable training
- ReLU activations
- Residual connections for deep networks
- Separate policy and value heads
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class ResidualBlock(nn.Module):
    """
    Residual block with two convolutional layers and skip connection.

    Architecture:
        x -> Conv -> BN -> ReLU -> Conv -> BN -> (+x) -> ReLU

    Args:
        channels: Number of input/output channels
    """

    def __init__(self, channels: int = 256):
        super().__init__()

        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)

        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with residual connection.

        Args:
            x: Input tensor of shape (batch, channels, 8, 8)

        Returns:
            Output tensor of same shape
        """
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = F.relu(out)

        return out


class PolicyHead(nn.Module):
    """
    Policy head that outputs move probabilities (AlphaZero paper architecture).

    Architecture:
        x -> Conv(3x3, 73 filters) -> BN -> ReLU -> Reshape(8x8x73) -> Flatten(4672)

    Following the AlphaZero paper, this outputs an 8x8x73 plane encoding 4672 moves:
    - 8x8: Starting square (from-square)
    - 73 planes per square:
      * 56 planes: Queen-style moves (8 directions × 7 distances)
        - N, NE, E, SE, S, SW, W, NW directions
        - 1-7 squares in each direction
      * 8 planes: Knight moves (L-shaped)
      * 9 planes: Underpromotions (3 directions × 3 piece types)
        - Left-diagonal, forward, right-diagonal
        - Promote to knight, bishop, or rook

    Note: Illegal moves are masked out and renormalized during MCTS search.

    Args:
        in_channels: Number of input channels from trunk (default 256)
    """

    def __init__(self, in_channels: int = 256):
        super().__init__()

        # AlphaZero uses 3x3 conv with 73 filters (not 1x1 conv)
        self.conv = nn.Conv2d(in_channels, 73, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(73)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass producing move probabilities.

        Args:
            x: Input tensor of shape (batch, in_channels, 8, 8)

        Returns:
            Policy logits of shape (batch, 4672) representing 8x8x73 move planes
        """
        out = self.conv(x)
        out = self.bn(out)
        out = F.relu(out)

        # Reshape from (batch, 73, 8, 8) to (batch, 8, 8, 73) to (batch, 4672)
        out = out.permute(0, 2, 3, 1)  # -> (batch, 8, 8, 73)
        out = out.reshape(out.size(0), -1)  # -> (batch, 4672)

        return out


class ValueHead(nn.Module):
    """
    Value head that outputs position evaluation.

    Architecture:
        x -> Conv(1x1) -> BN -> ReLU -> Flatten -> FC(256) -> ReLU -> FC(1) -> Tanh

    Outputs a scalar in [-1, 1] representing position evaluation:
        +1 = winning for current player
        -1 = losing for current player
         0 = drawn position

    Args:
        in_channels: Number of input channels from trunk
    """

    def __init__(self, in_channels: int = 256):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(1)

        # 1 channel * 8 * 8 = 64 features
        self.fc1 = nn.Linear(64, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass producing position evaluation.

        Args:
            x: Input tensor of shape (batch, in_channels, 8, 8)

        Returns:
            Value tensor of shape (batch, 1) in range [-1, 1]
        """
        out = self.conv(x)
        out = self.bn(out)
        out = F.relu(out)

        out = out.view(out.size(0), -1)  # Flatten

        out = self.fc1(out)
        out = F.relu(out)

        out = self.fc2(out)
        out = torch.tanh(out)

        return out


class AlphaZeroNet(nn.Module):
    """
    AlphaZero neural network for chess (following the original paper).

    This network takes a board representation as input and outputs:
    1. Policy: Probability distribution over 4672 moves (8x8x73 encoding)
    2. Value: Position evaluation from current player's perspective [-1, 1]

    Architecture (from AlphaZero paper):
        Input (119 planes) -> Conv Block (256 filters, 3x3)
        -> 19 Residual Blocks (256 filters, 3x3)
        -> Policy Head (73 filters, 3x3 -> 4672 moves)
        -> Value Head (1 filter -> FC 256 -> FC 1)

    Input representation (119 planes per 8x8 square):
        - 6 piece types × 2 colors × 8 history positions = 96 planes
        - 2 planes for repetition counts
        - 4 planes for castling rights
        - 1 plane for side to move
        - 1 plane for move count
        - 15 planes for no-progress count
        Total: 119 planes

    Args:
        input_channels: Number of input planes (default 119)
        num_res_blocks: Number of residual blocks (default 19)
        channels: Number of channels in residual tower (default 256)

    Example:
        >>> # Full AlphaZero network
        >>> net = AlphaZeroNet(input_channels=119, num_res_blocks=19, channels=256)
        >>> board_state = torch.randn(32, 119, 8, 8)  # batch of 32
        >>> policy, value = net(board_state)
        >>> print(policy.shape, value.shape)
        torch.Size([32, 4672]) torch.Size([32, 1])
    """

    def __init__(
        self,
        input_channels: int = 119,
        num_res_blocks: int = 19,
        channels: int = 256
    ):
        super().__init__()

        # Initial convolutional block (3x3 conv with 256 filters)
        self.input_conv = nn.Conv2d(input_channels, channels, kernel_size=3, padding=1, bias=False)
        self.input_bn = nn.BatchNorm2d(channels)

        # Residual tower (19 blocks × 2 conv layers each)
        # Use nn.ModuleDict to match naming convention: residual.0, residual.1, etc.
        # This is important for federated learning aggregation logic
        self.residual = nn.ModuleDict({
            str(i): ResidualBlock(channels) for i in range(num_res_blocks)
        })

        # Policy and value heads
        self.policy_head = PolicyHead(channels)
        self.value_head = ValueHead(channels)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.

        Args:
            x: Input board state tensor of shape (batch, 119, 8, 8)

        Returns:
            Tuple of (policy_logits, value):
                - policy_logits: Shape (batch, 4672) - raw logits over 8x8x73 move planes
                - value: Shape (batch, 1) - position evaluation in [-1, 1]
        """
        # Initial convolution
        out = self.input_conv(x)
        out = self.input_bn(out)
        out = F.relu(out)

        # Residual tower - iterate through ModuleDict in order
        for i in sorted(self.residual.keys(), key=int):
            out = self.residual[i](out)

        # Dual heads
        policy = self.policy_head(out)
        value = self.value_head(out)

        return policy, value

    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict with softmax applied to policy.

        Args:
            x: Input board state tensor

        Returns:
            Tuple of (policy_probs, value):
                - policy_probs: Shape (batch, policy_size) - move probabilities (sums to 1)
                - value: Shape (batch, 1) - position evaluation in [-1, 1]
        """
        self.eval()
        with torch.no_grad():
            policy_logits, value = self.forward(x)
            policy_probs = F.softmax(policy_logits, dim=1)
        return policy_probs, value


def create_alphazero_net(
    input_channels: int = 119,
    num_res_blocks: int = 19,
    channels: int = 256
) -> AlphaZeroNet:
    """
    Factory function to create an AlphaZero network.

    Provides a convenient way to create networks with different sizes:
    - Tiny (for testing): num_res_blocks=2, channels=64
    - Small (for quick experiments): num_res_blocks=5, channels=128
    - Medium (for laptops): num_res_blocks=10, channels=256
    - Full (AlphaZero paper): num_res_blocks=19, channels=256
    - Large (AlphaZero final): num_res_blocks=40, channels=256

    All networks output 4672 moves (8x8x73 encoding) as per AlphaZero paper.

    Args:
        input_channels: Number of input planes (default 119)
        num_res_blocks: Number of residual blocks (default 19)
        channels: Number of channels in residual tower (default 256)

    Returns:
        Initialized AlphaZeroNet

    Example:
        >>> # Small network for testing
        >>> small_net = create_alphazero_net(num_res_blocks=5, channels=128)
        >>>
        >>> # Full AlphaZero network (19 blocks, 256 channels)
        >>> full_net = create_alphazero_net(num_res_blocks=19, channels=256)
        >>>
        >>> # Final AlphaZero (40 blocks)
        >>> large_net = create_alphazero_net(num_res_blocks=40, channels=256)
    """
    return AlphaZeroNet(
        input_channels=input_channels,
        num_res_blocks=num_res_blocks,
        channels=channels
    )


# if __name__ == "__main__":
#     # Test the network
#     print("Testing AlphaZero Network...")

#     # Create a small network for testing
#     net = create_alphazero_net(input_channels=119, num_res_blocks=5, channels=128)

#     # Print network info
#     total_params = sum(p.numel() for p in net.parameters())
#     trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
#     print(f"\nNetwork Parameters:")
#     print(f"  Total: {total_params:,}")
#     print(f"  Trainable: {trainable_params:,}")

#     # Test forward pass
#     batch_size = 4
#     dummy_input = torch.randn(batch_size, 119, 8, 8)

#     print(f"\nTesting forward pass with batch_size={batch_size}...")
#     policy, value = net(dummy_input)

#     print(f"  Policy shape: {policy.shape}")
#     print(f"  Value shape: {value.shape}")
#     print(f"  Value range: [{value.min().item():.3f}, {value.max().item():.3f}]")

#     # Test predict method
#     print(f"\nTesting predict method...")
#     policy_probs, value = net.predict(dummy_input)
#     print(f"  Policy probs sum: {policy_probs.sum(dim=1)}")
#     print(f"  Policy probs range: [{policy_probs.min().item():.6f}, {policy_probs.max().item():.6f}]")

#     print("\nNetwork test complete!")
