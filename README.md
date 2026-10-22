# Draughts Positions Model

A MobileNetV2 model for classifying draughts-board squares.

## Requirements

- Python
- The packages listed in [`requirements.txt`](requirements.txt)

## Setup

Run these commands from the project root in PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If PowerShell does not allow activating the virtual environment, use its
interpreter directly instead:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Prepare the dataset

1. Put source board images (`.png` or `.jpg`) in `raw_data/`.
2. Run the preprocessing script from the project root:

   ```powershell
   python src/preprocess.py
   ```

3. The script resizes each board and saves crops of its dark squares to
   `dataset/unlabeled/`.
4. Move each crop from `dataset/unlabeled/` into the matching class folder:

   ```text
   dataset/
   ├── black_checker/
   ├── black_king/
   ├── empty/
   ├── white_checker/
   └── white_king/
   ```

   After moving the crops, remove the now-empty `dataset/unlabeled/` folder.
   The training script treats subfolders under `dataset/` as classes.

## Train the model

From the project root, run:

```powershell
python src/train.py
```

The script trains for 20 epochs and saves the Keras model as
`checkers_mobilenetv2.h5` in the project root.

## Export to ONNX (optional)

Install `tf2onnx` in the active virtual environment:

```powershell
python -m pip install tf2onnx
```

Then convert the saved Keras model:

```powershell
python -m tf2onnx.convert --keras checkers_mobilenetv2.h5 --output export_model/model.onnx
```

The ONNX model will be written to `export_model/model.onnx`.
