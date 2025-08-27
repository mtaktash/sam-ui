# Manual labelling and propagation with SAM-2

## Environment creation
```
conda create -n sam2 python=3.12
conda activate sam2
pip install opencv-python matplotlib scipy
pip install 'git+https://github.com/facebookresearch/sam2.git'
pip install huggingface-hub
```

## Tracking GUI
```
python scripts/tracking_gui.py --frames-path %path/to/frames% 
```

Availiable arguments
- `--output-path` -- Path to the output directory
- `--ui-scale` -- Scale factor for the UI (in case the window is too big or small)
- `--clear-output` -- If provided, the script will clear the output directory before starting"

## GUI usage

- Arrow Down ⬇️ → Select previous object
- Arrow Up ⬆️ → Select next object
- Arrow Left ⬅️ → Go to previous frame
- Arrow Right ➡️ → Go to next frame
- P → Propagate changes to next frames (default is 16 frames forward)
- R → Reset state completely
- S → Save all progress
- C → Clear output

## GUI usage video