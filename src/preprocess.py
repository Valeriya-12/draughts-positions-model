import cv2
import numpy as np
import os
from pathlib import Path

RAW_DATA_DIR = Path("raw_data")
DATASET_DIR = Path("dataset")
CELL_SIZE = 64  # Entered size for MobileNetV2

def setup_dataset_structure():
    # Create folders for dataset if they don't exist
    classes = ['empty', 'white_checker', 'black_checker', 'white_king', 'black_king']
    for cls in classes:
        os.makedirs(DATASET_DIR / cls, exist_ok=True)

def remove_hatching(cell_img: np.ndarray) -> np.ndarray:
    # Remove diagonal hatching or noise using morphological operations
    # Simple diagonal kernel to close diagonal lines
    kernel_diag = np.array([
        [0, 0, 1],
        [0, 1, 0],
        [1, 0, 0]
    ], dtype=np.uint8)

    # Morphological closing fills in thin strokes
    closed = cv2.morphologyEx(cell_img, cv2.MORPH_CLOSE, kernel_diag, iterations=2)

    # A median filter removes any remaining striped noise
    denoised = cv2.medianBlur(closed, ksize=5)

    return denoised

def process_board_image(image_path: Path):
    # Load a diagram, straighten the board, filter its cells, and crop them
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Failed to load image: {image_path}")
        return

    # Resize the board to 512x512 (8 cells * 64 px).
    board_dim = 8 * CELL_SIZE
    resized_board = cv2.resize(img, (board_dim, board_dim))

    img_name = image_path.stem

    for row in range(8):
        for col in range(8):
            # Draughts is played on dark squares: (row + col) % 2 == 1
            if (row + col) % 2 == 1:
                y_start, y_end = row * CELL_SIZE, (row + 1) * CELL_SIZE
                x_start, x_end = col * CELL_SIZE, (col + 1) * CELL_SIZE

                raw_cell = resized_board[y_start:y_end, x_start:x_end]
                clean_cell = remove_hatching(raw_cell)

                # Save cells to the temporary 'unlabeled' folder for labeling
                unlabeled_dir = DATASET_DIR / "unlabeled"
                os.makedirs(unlabeled_dir, exist_ok=True)

                out_filename = f"{img_name}_r{row}_c{col}.png"
                cv2.imwrite(str(unlabeled_dir / out_filename), clean_cell)

def main():
    setup_dataset_structure()

    if not RAW_DATA_DIR.exists():
        os.makedirs(RAW_DATA_DIR)
        print(f"Created {RAW_DATA_DIR}. Place the original board images there.")
        return

    image_files = list(RAW_DATA_DIR.glob("*.png")) + list(RAW_DATA_DIR.glob("*.jpg"))

    for img_path in image_files:
        print(f"Processing: {img_path.name}")
        process_board_image(img_path)

    print("\nCropping complete. Processed cells are in dataset/unlabeled/.")
    print("Move the images into the corresponding folders: empty, white_pawn, black_pawn.")

if __name__ == "__main__":
    main()
