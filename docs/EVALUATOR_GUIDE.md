# Model Evaluator Guide - Test Your Models vs Stockfish 🎯

Evaluate your trained federated learning models by playing them against Stockfish to estimate their ELO rating.

## Quick Start

### Install Stockfish

```bash
# Ubuntu/Debian
sudo apt install stockfish

# MacOS
brew install stockfish

# Check installation
which stockfish
```

### Basic Usage

```bash
cd /home/fra/Uni/Thesis/main_repo/FedRL/chess-federated-learning

# Auto-evaluate (recommended for first test)
uv run python evaluator.py --model storage/models/cluster_tactical_round_5.pt --auto --games 10

# Test against specific ELO
uv run python evaluator.py --model storage/models/cluster_tactical_round_5.pt --elo 1200 --games 20

# Save results to file
uv run python evaluator.py --model storage/models/cluster_tactical_round_5.pt --auto --output results.json
```

## Usage Examples

### 1. Quick Evaluation (10 games)
```bash
uv run python evaluator.py \
    --model storage/models/cluster_tactical_round_5.pt \
    --auto \
    --games 10 \
    --start-elo 1200
```

**Output:**
- Plays 10 games each at ELO 1000, 1200, 1400
- Estimates model's ELO rating
- Shows win/draw/loss statistics

### 2. Thorough Evaluation (30 games per level)
```bash
uv run python evaluator.py \
    --model storage/models/cluster_positional_round_10.pt \
    --auto \
    --games 30 \
    --start-elo 1500 \
    --output eval_results.json
```

### 3. Test Against Specific Opponent
```bash
# Test if your model can beat ELO 1000
uv run python evaluator.py \
    --model storage/models/cluster_tactical_round_5.pt \
    --elo 1000 \
    --games 20
```

### 4. Compare Multiple Models
```bash
# Evaluate tactical model
uv run python evaluator.py --model storage/models/cluster_tactical_round_5.pt --auto --output tactical_eval.json

# Evaluate positional model
uv run python evaluator.py --model storage/models/cluster_positional_round_5.pt --auto --output positional_eval.json

# Compare results
cat tactical_eval.json | grep estimated_elo
cat positional_eval.json | grep estimated_elo
```

## Options

### Required
- `--model PATH` - Path to model checkpoint file

### Optional
- `--auto` - Auto-evaluate across multiple ELO levels (recommended)
- `--elo N` - Test against specific Stockfish ELO (800-2800)
- `--games N` - Number of games per level (default: 20)
- `--start-elo N` - Starting ELO guess for auto mode (default: 1200)
- `--stockfish PATH` - Path to Stockfish binary (default: /usr/bin/stockfish)
- `--output FILE` - Save results to JSON file
- `--time SECONDS` - Time per move (default: 0.1s)
- `--device {cpu,cuda}` - Device to run model on
- `--quiet` - Less verbose output

## Understanding Results

### Win Rates & ELO
- **>80% wins** = Model is much stronger (play higher ELO)
- **50-80% wins** = Model is slightly stronger
- **40-60%** = Even match (close to opponent's ELO)
- **20-40%** = Model is weaker
- **<20% wins** = Much weaker (play lower ELO)

### ELO Ranges
- **800-1000**: Beginner (knows rules, random play)
- **1000-1200**: Novice (basic tactics)
- **1200-1400**: Intermediate (understands opening/endgame basics)
- **1400-1600**: Club player (solid tactics)
- **1600-1800**: Strong club player
- **1800-2000**: Expert
- **2000+**: Master level

### Typical Results
After 5-10 training rounds, expect:
- **Tactical model**: 900-1200 ELO
- **Positional model**: 800-1100 ELO
- **Aggregated model**: 1000-1300 ELO

After 20+ rounds with good data:
- **Well-trained models**: 1200-1500 ELO
- **Highly-trained models**: 1500-1800 ELO

## Output Format

### Console Output
```
Match: AlphaZero-FL vs Stockfish (ELO 1200)
Games: 20, Time per move: 0.1s

Game 1/20:
  AlphaZero-FL (White) vs Stockfish-1200 (Black)
  Result: 1-0 (checkmate) in 45 moves (4.5s)

...

Match Result: 12W-3D-5L
Score: 13.5/20 (67.5%)

ESTIMATED ELO: 1325 ± 100
```

### JSON Output (`--output results.json`)
```json
{
  "estimated_elo": 1325,
  "confidence_range": 100,
  "matches": [
    {
      "opponent": "Stockfish-1200",
      "opponent_elo": 1200,
      "games_played": 20,
      "wins": 12,
      "draws": 3,
      "losses": 5,
      "score": 13.5,
      "win_rate": 67.5
    }
  ],
  "timestamp": "2025-10-14T17:30:00"
}
```

## Advanced Usage

### Test Against Multiple ELO Levels
```bash
# Test weak opponent
uv run python evaluator.py --model mymodel.pt --elo 800 --games 10

# Test medium opponent
uv run python evaluator.py --model mymodel.pt --elo 1400 --games 10

# Test strong opponent
uv run python evaluator.py --model mymodel.pt --elo 2000 --games 10
```

### Longer Time Controls
```bash
# Give each player 1 second per move (more accurate)
uv run python evaluator.py --model mymodel.pt --auto --time 1.0

# Blitz-style (0.5s per move)
uv run python evaluator.py --model mymodel.pt --auto --time 0.5
```

### GPU Acceleration
```bash
# Use GPU if available
uv run python evaluator.py --model mymodel.pt --auto --device cuda
```

## Troubleshooting

### "Stockfish not found"
```bash
# Install Stockfish
sudo apt install stockfish

# Or specify custom path
uv run python evaluator.py --model mymodel.pt --auto --stockfish /path/to/stockfish
```

### "Failed to load model"
- Check model path is correct
- Ensure model was saved from training
- Try a model from a later round

### Model plays poorly
- Early training rounds (1-3) will be weak
- Need more training data
- Check if model loaded correctly

### Evaluation takes too long
- Reduce `--games` (try 10 instead of 20)
- Use faster time control `--time 0.05`
- Use `--quiet` to reduce logging

## Interpreting Your Model's Strength

### After 5 Rounds
If your model achieves:
- **ELO 900-1100**: Good! Basic understanding
- **ELO 1100-1300**: Very good! Solid tactics
- **ELO 1300+**: Excellent! Strong player

### After 10 Rounds
Target:
- **ELO 1200-1400**: Expected with good data
- **ELO 1400-1600**: Very strong
- **ELO 1600+**: Exceptionally well-trained

### After 20+ Rounds
With sufficient training:
- **ELO 1500-1700**: Master-level play
- **ELO 1700-2000**: Expert level
- **ELO 2000+**: Approaching engine strength

## Tips for Better Evaluation

1. **Play more games**: 10 games minimum, 20-30 better, 50+ for precision
2. **Test multiple levels**: Auto mode tests 3 levels automatically
3. **Allow thinking time**: Use `--time 1.0` for more accurate play
4. **Save results**: Always use `--output` to track progress
5. **Compare rounds**: Evaluate models from different training rounds

## Example Workflow

```bash
# 1. Evaluate early model
uv run python evaluator.py --model storage/models/cluster_tactical_round_3.pt \
    --auto --games 15 --output eval_round3.json

# 2. Evaluate later model
uv run python evaluator.py --model storage/models/cluster_tactical_round_10.pt \
    --auto --games 15 --output eval_round10.json

# 3. Compare improvement
echo "Round 3:" && cat eval_round3.json | grep estimated_elo
echo "Round 10:" && cat eval_round10.json | grep estimated_elo
```

## Performance Notes

- Each game takes ~2-10 seconds depending on time control
- 20 games at 3 ELO levels = ~5-10 minutes total
- GPU doesn't help much (Stockfish is CPU-bound)
- Evaluation is deterministic with same settings

---

**Ready to test your model?** Start with:
```bash
uv run python evaluator.py --model storage/models/<your_model>.pt --auto
```

🎯 Good luck!
