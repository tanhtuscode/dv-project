# SkateboardML - Binary Classification Project

## Project Focus
This project has been focused on binary classification of skateboarding tricks:
- **Ollie**: Basic skateboard jump
- **Kickflip**: Ollie with board flip

## Data Structure
```
Tricks/
  Kickflip/          # Kickflip video and feature files
  Ollie/             # Ollie video and feature files

trainlist_binary.txt  # Training file list (Ollie + Kickflip only)
testlist_binary.txt   # Test file list (Ollie + Kickflip only)
```

## Updated Scripts
- `train_binary.py`: Binary classification training
- `train_windows.py`: Updated for binary classification
- `MLScript.py`: Updated for binary classification

## Next Steps
1. Run `python train_binary.py` for training
2. Use `scripts/generate_charts.py` for visualization
3. Deploy with `app.py` Flask application
