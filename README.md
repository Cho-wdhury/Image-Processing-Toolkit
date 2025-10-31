# 🖼️ Image Processing Toolkit (Manual NumPy)

A desktop application built with **Python**, **Tkinter**, **NumPy**, and **Pillow (PIL)** to perform classic image-processing operations implemented manually with NumPy — no OpenCV required.

---

## ✨ Features

- Open, preview, process, and save images (PNG, JPG/JPEG, BMP, TIFF)
- Side-by-side **Original** vs **Output** preview
- **Undo / Redo / Reset**
- Built-in **OceanWave** theme (modern dark UI)
- Editable **Developer Badge** (name, ID, photo)
- **Algorithms implemented manually:**
  - Negative
  - Log Transform / Gamma Correction
  - Box & Gaussian Smoothing
  - Unsharp Mask Sharpening
  - Manual / Otsu Thresholding
  - Sobel Edge Detection
  - Histogram (Grayscale)
  - Resize (Nearest & Bilinear)

---

## 🧠 Algorithms & Files

| Category | Functions | File |
|-----------|------------|------|
| Negative | `negative()` | `algorithms/negative.py` |
| Log & Gamma | `log_transform()`, `gamma_transform()` | `algorithms/log_gamma.py` |
| Smoothing | `smooth_box()`, `smooth_gaussian()` | `algorithms/smoothing.py` |
| Sharpening | `unsharp_mask()` | `algorithms/sharpening.py` |
| Histogram | `histogram_gray()` | `algorithms/histrogram.py` *(keep the original typo)* |
| Resize | `resize_nearest()`, `resize_bilinear()` | `algorithms/resize.py` |
| Threshold | `threshold_apply()`, `otsu_threshold()` | `algorithms/threshold.py` |
| Edges | `sobel_edges()` | `algorithms/edges.py` |

---

## 📁 Project Structure

.
├── main.py
├── algorithms/
│ ├── negative.py
│ ├── log_gamma.py
│ ├── smoothing.py
│ ├── sharpening.py
│ ├── histrogram.py
│ ├── resize.py
│ ├── threshold.py
│ └── edges.py
├── developer_info.json # saved developer name/ID/photo
├── nishat.jpg # default developer image
└── README.md

yaml
Copy code

---

## ⚙️ Requirements

```bash
pip install numpy pillow
Tkinter usually comes with Python.
On Linux, if missing:
sudo apt-get install python3-tk

▶️ How to Run
bash
Copy code
python main.py
Keyboard Shortcuts:

Ctrl+O → Open Image

Ctrl+S → Save Output

Ctrl+Z → Undo

Ctrl+Y → Redo

🧰 Building Executable (PyInstaller)
The app includes resource_path() for bundled builds.

bash
Copy code
pyinstaller --onefile --windowed main.py
To include extra files (like the developer photo):

bash
Copy code
pyinstaller --onefile --windowed \
  --add-data "nishat.jpg;." \
  main.py
👩‍💻 Developer Info
Name: Nishat Tasnim Chowdhury

ID: 0812220105101022

Click the photo area in the app to edit your name, ID, and photo.
The info is saved in developer_info.json.

🎨 OceanWave Theme
Element	Color
Background	#0E1B25
Panel	#132632
Border	#1f3b4d
Text	#EAF6FF
Subtle Text	#9cc6d8
Blue	#2FA4F6
Teal	#25C9B7
Orange	#FFB020
Red	#F44336

🧩 Usage Example
Open an image.

Choose a transformation (e.g. Smoothing, Gamma, Threshold).

Adjust parameters and apply.

Compare results on the Output panel.

Save your processed image.

🐞 Troubleshooting
Issue	Solution
ModuleNotFoundError: histrogram	Keep filename as histrogram.py
Tkinter not found	Install with sudo apt-get install python3-tk
Output not saving	Apply at least one operation before saving

📜 License
MIT License © 2025 — Nishat Tasnim Chowdhury

yaml
Copy code
---

✅ Just copy everything above into your `README.md` file on GitHub — it’s perfectly formatted and will render beautifully.






