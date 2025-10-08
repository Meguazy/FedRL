# AlphaZero Move Encoding (8x8x73 = 4672 moves)

This document describes the move encoding scheme used in AlphaZero for chess.

## Overview

The policy head outputs an **8×8×73** tensor representing **4,672 possible moves**:
- **8×8**: Starting square (from-square) on the chess board
- **73 planes**: Different move types from each square

## The 73 Move Planes

For each of the 64 starting squares, there are 73 possible move types:

### 1. Queen-style moves (56 planes)

These represent sliding moves in 8 directions, up to 7 squares away:

- **8 directions**: N, NE, E, SE, S, SW, W, NW
- **7 distances**: 1, 2, 3, 4, 5, 6, or 7 squares
- **Total**: 8 × 7 = 56 planes

```
Plane 0-6:   North (1-7 squares)
Plane 7-13:  NorthEast (1-7 squares)
Plane 14-20: East (1-7 squares)
Plane 21-27: SouthEast (1-7 squares)
Plane 28-34: South (1-7 squares)
Plane 35-41: SouthWest (1-7 squares)
Plane 42-48: West (1-7 squares)
Plane 49-55: NorthWest (1-7 squares)
```

These planes cover:
- All queen moves
- All rook moves (N, E, S, W only)
- All bishop moves (NE, SE, SW, NW only)
- All king moves (distance 1 only)
- Normal pawn moves (N or S, distance 1 or 2)

### 2. Knight moves (8 planes)

Knight moves in L-shape (2 squares in one direction, 1 square perpendicular):

```
Plane 56: Knight move NNE (2 North, 1 East)
Plane 57: Knight move ENE (2 East, 1 North)
Plane 58: Knight move ESE (2 East, 1 South)
Plane 59: Knight move SSE (2 South, 1 East)
Plane 60: Knight move SSW (2 South, 1 West)
Plane 61: Knight move WSW (2 West, 1 South)
Plane 62: Knight move WNW (2 West, 1 North)
Plane 63: Knight move NNW (2 North, 1 West)
```

### 3. Underpromotions (9 planes)

Pawn promotions to pieces other than queen (queen promotion uses queen-style move planes):

- **3 directions**: Left-diagonal, Forward, Right-diagonal
- **3 piece types**: Knight, Bishop, Rook
- **Total**: 3 × 3 = 9 planes

```
Plane 64: Promote to Knight, move left-diagonal
Plane 65: Promote to Knight, move forward
Plane 66: Promote to Knight, move right-diagonal
Plane 67: Promote to Bishop, move left-diagonal
Plane 68: Promote to Bishop, move forward
Plane 69: Promote to Bishop, move right-diagonal
Plane 70: Promote to Rook, move left-diagonal
Plane 71: Promote to Rook, move forward
Plane 72: Promote to Rook, move right-diagonal
```

## Move Representation

A move is represented by:
1. **From-square**: (row, col) in range [0, 7]
2. **Plane index**: Which of the 73 move types in range [0, 72]

Converting to flat index: `index = row * 8 * 73 + col * 73 + plane`

## Handling Illegal Moves

The network outputs probabilities for all 4,672 moves, including many illegal ones.

During MCTS search:
1. Get policy output from network
2. Mask out illegal moves (set probability to 0)
3. Renormalize remaining legal moves

Example:
```python
# Get raw policy
policy_logits = net(board)[0]  # Shape: (4672,)

# Get legal moves for current position
legal_moves = get_legal_moves(board)  # List of legal move indices

# Create mask
mask = torch.zeros(4672)
mask[legal_moves] = 1.0

# Apply mask and renormalize
masked_policy = policy_logits * mask
masked_policy = F.softmax(masked_policy, dim=0)
```

## Implementation Notes

### PyTorch Network Output

The policy head outputs shape `(batch, 4672)`:
```python
# In PolicyHead.forward():
out = self.conv(x)  # -> (batch, 73, 8, 8)
out = out.permute(0, 2, 3, 1)  # -> (batch, 8, 8, 73)
out = out.reshape(batch, -1)  # -> (batch, 4672)
```

### Decoding a Move

To convert a policy index back to a chess move:
```python
def decode_move(index):
    """Decode flat index to (from_row, from_col, plane)"""
    from_row = index // (8 * 73)
    from_col = (index % (8 * 73)) // 73
    plane = index % 73
    return from_row, from_col, plane

def plane_to_move_type(plane):
    """Convert plane index to move description"""
    if plane < 56:  # Queen-style
        direction = plane // 7
        distance = (plane % 7) + 1
        directions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
        return f"{directions[direction]}-{distance}"
    elif plane < 64:  # Knight
        knights = ['NNE', 'ENE', 'ESE', 'SSE', 'SSW', 'WSW', 'WNW', 'NNW']
        return f"Knight-{knights[plane - 56]}"
    else:  # Underpromotion
        p = plane - 64
        piece = ['Knight', 'Bishop', 'Rook'][p // 3]
        direction = ['Left', 'Forward', 'Right'][p % 3]
        return f"Promote-{piece}-{direction}"
```

## References

- AlphaZero paper: "Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm"
- Move encoding is described in the paper's supplementary materials
