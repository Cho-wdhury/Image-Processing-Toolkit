import os, sys, json, time, tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter import font as tkfont
from typing import Optional

import numpy as np
from PIL import Image, ImageTk

# === algorithms (your exact filenames) ===
from algorithms.negative import negative
from algorithms.log_gamma import log_transform, gamma_transform
from algorithms.smoothing import smooth_box, smooth_gaussian
from algorithms.sharpening import unsharp_mask
from algorithms.histrogram import histogram_gray   # (file named histrogram.py)
from algorithms.resize import resize_nearest, resize_bilinear
from algorithms.threshold import threshold_apply, otsu_threshold
from algorithms.edges import sobel_edges


# ---------- helper for PyInstaller one-file resources ----------
def resource_path(rel_path: str) -> str:
    """Get absolute path to resource for dev & PyInstaller one-file exe."""
    try:
        base = sys._MEIPASS  # set by PyInstaller at runtime
    except AttributeError:
        base = os.path.abspath(".")
    return os.path.join(base, rel_path)


# ---------- PIL <-> NumPy helpers ----------
def pil_to_array(im: Image.Image) -> np.ndarray:
    if im.mode in ("L", "I;16", "I"):
        return np.array(im.convert("L"), dtype=np.uint8)
    return np.array(im.convert("RGB"), dtype=np.uint8)

def array_to_pil(arr: np.ndarray) -> Image.Image:
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        return Image.fromarray(arr, mode="L")
    elif arr.ndim == 3 and arr.shape[2] == 3:
        return Image.fromarray(arr, mode="RGB")
    raise ValueError("Unsupported array shape")


class ImageToolkitApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Image Processing Toolkit (Manual NumPy)")
        self.geometry("1120x680")
        self.minsize(1080, 640)

        # state
        self.img_open: Optional[Image.Image] = None
        self.img: Optional[Image.Image] = None
        self.undo_stack, self.redo_stack = [], []
        self.open_filename = None

        self._button_refs = []  # (btn, color_key)

        self._init_theme()
        self._build_menu()
        self._build_layout()
        self._load_developer_info()

    # ---------- Single (OceanWave) theme ----------
    def _init_theme(self):
        self._apply_palette()

    def _apply_palette(self):
        # OceanWave palette only (no theme switch visible)
        self.colors = dict(
            bg="#0E1B25", panel="#132632", border="#1f3b4d",
            fg="#EAF6FF", subtle="#9cc6d8",
            blue="#2FA4F6", teal="#25C9B7", orange="#FFB020",
            red="#F44336", gray="#616D7A", canvas="#0A0F15",
        )
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except Exception:
            pass

        self.configure(bg=self.colors["bg"])
        tkfont.nametofont("TkDefaultFont").configure(size=10)
        tkfont.nametofont("TkHeadingFont").configure(size=11, weight="bold")

        style.configure(".", background=self.colors["bg"], foreground=self.colors["fg"])
        style.configure("TFrame", background=self.colors["bg"])
        style.configure("TLabel", background=self.colors["bg"], foreground=self.colors["fg"])
        style.configure("Subtle.TLabel", background=self.colors["bg"], foreground=self.colors["subtle"])
        style.configure("Title.TLabel", background=self.colors["bg"], foreground=self.colors["fg"],
                        font=("Segoe UI", 10, "bold"))

        # Dialog input clarity
        style.configure("Entry.TEntry", fieldbackground=self.colors["panel"], foreground=self.colors["fg"])
        style.configure("Cmb.TCombobox",
                        fieldbackground=self.colors["panel"],
                        background=self.colors["panel"],
                        foreground=self.colors["fg"])
        style.map("Cmb.TCombobox",
                  fieldbackground=[("readonly", self.colors["panel"])],
                  foreground=[("readonly", self.colors["fg"])])

        # Nice visible scales
        style.configure("Blue.Horizontal.TScale",
                        troughcolor=self.colors["panel"],
                        background=self.colors["blue"])
        style.configure("Gray.Horizontal.TScale",
                        troughcolor=self.colors["panel"],
                        background=self.colors["subtle"])

        # recolor buttons/canvases if already built
        for btn, key in getattr(self, "_button_refs", []):
            try:
                btn.configure(bg=self.colors[key],
                              activebackground=self._brighten(self.colors[key]))
            except Exception:
                pass
        for canv in getattr(self, "_canvases", []):
            canv.configure(bg=self.colors["canvas"],
                           highlightbackground=self.colors["border"])

    # ---------- Menus ----------
    def _build_menu(self):
        menubar = tk.Menu(self)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Open...", command=self.open_image, accelerator="Ctrl+O")
        file_menu.add_command(label="Save Output Image...", command=self.save_as, accelerator="Ctrl+S")
        file_menu.add_separator()
        file_menu.add_command(label="Undo", command=self.undo, accelerator="Ctrl+Z")
        file_menu.add_command(label="Redo", command=self.redo, accelerator="Ctrl+Y")
        file_menu.add_command(label="Reset", command=self.reset_image)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.quit)
        menubar.add_cascade(label="File", menu=file_menu)
        self.config(menu=menubar)

        self.bind_all("<Control-o>", lambda e: self.open_image())
        self.bind_all("<Control-s>", lambda e: self.save_as())
        self.bind_all("<Control-z>", lambda e: self.undo())
        self.bind_all("<Control-y>", lambda e: self.redo())

    # ---------- Layout ----------
    def _build_layout(self):
        # rows: header, toolbar, titles, canvases, status
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=0)
        self.rowconfigure(1, weight=0)
        self.rowconfigure(2, weight=0)
        self.rowconfigure(3, weight=1)
        self.rowconfigure(4, weight=0)

        # ===== Header =====
        header = ttk.Frame(self)
        header.grid(row=0, column=0, sticky="ew", padx=12, pady=(10, 8))
        header.columnconfigure(0, weight=1)
        header.columnconfigure(1, weight=0)

        # left title
        left_head = ttk.Frame(header)
        left_head.grid(row=0, column=0, sticky="w")
        ttk.Label(left_head, text="Image Processing Toolkit",
                  font=("Segoe UI", 16, "bold")).pack(anchor="w")

        # right badge (photo + name with ID directly below, no gap)
        badge = ttk.Frame(header)
        badge.grid(row=0, column=1, sticky="e")
        self.dev_name_var = tk.StringVar(value="Nishat Tasnim Chowdhury")
        self.dev_id_var = tk.StringVar(value="ID: 0812220105101022")

        self.dev_photo_label = ttk.Label(badge, cursor="hand2")
        self.dev_photo_label.grid(row=0, column=0, rowspan=2, padx=(0, 10), sticky="n")
        self.dev_photo_label.bind("<Button-1>", lambda e: self._ask_developer_info())

        name_id_frame = ttk.Frame(badge)
        name_id_frame.grid(row=0, column=1, sticky="e")
        ttk.Label(name_id_frame, textvariable=self.dev_name_var,
                  font=("Segoe UI", 12, "bold")).grid(row=0, column=0, sticky="e")
        ttk.Label(name_id_frame, textvariable=self.dev_id_var,
                  font=("Segoe UI", 11)).grid(row=1, column=0, sticky="e")

        # ===== Toolbar (3 rows) =====
        self.toolbar = tk.Frame(self, bg=self.colors["bg"])
        self.toolbar.grid(row=1, column=0, sticky="ew", padx=12, pady=(0, 8))
        for c in range(5):
            self.toolbar.grid_columnconfigure(c, weight=1)

        def tbtn(color_key, text, cmd, r, c):
            b = tk.Button(self.toolbar, text=text, command=cmd,
                          bg=self.colors[color_key], fg="white",
                          bd=0, relief=tk.FLAT,
                          highlightthickness=0, takefocus=0,
                          activebackground=self._brighten(self.colors[color_key]),
                          activeforeground="white",
                          font=("Segoe UI", 10, "bold"), padx=12, pady=6, cursor="hand2")
            b.grid(row=r, column=c, padx=6, pady=4, sticky="ew")
            self._button_refs.append((b, color_key))
            return b

        # Row 1
        tbtn("gray", "Open Image", self.open_image, 0, 0)
        tbtn("red", "Reset", self.reset_image, 0, 1)
        tbtn("teal", "Save Output Image", self.save_as, 0, 2)

        # Row 2
        tbtn("blue", "Negative",
             lambda: self._apply_simple(negative, "Negative"), 1, 0)
        tbtn("blue", "Smoothing", self._dlg_smoothing, 1, 1)
        tbtn("blue", "Sharpen", self._dlg_sharpen, 1, 2)
        tbtn("blue", "Threshold", self._dlg_threshold, 1, 3)
        tbtn("blue", "Edges",
             lambda: self._apply_simple(sobel_edges, "Sobel Edges"), 1, 4)

        # Row 3
        tbtn("blue", "Histogram", self.show_histogram, 2, 0)
        tbtn("orange", "Log Transform",
             lambda: self._apply_simple(log_transform, "Log Transform"), 2, 1)
        tbtn("orange", "Gamma Transform", self._dlg_gamma, 2, 2)
        tbtn("gray", "Resize", self._dlg_resize, 2, 3)

        # ===== Titles =====
        titles = ttk.Frame(self)
        titles.grid(row=2, column=0, sticky="ew", padx=12, pady=(0, 6))
        titles.columnconfigure(0, weight=1)
        titles.columnconfigure(1, weight=1)

        self.orig_title_var = tk.StringVar(value="Original Image")
        self.proc_title_var = tk.StringVar(value="Output Image")

        ttk.Label(titles, textvariable=self.orig_title_var,
                  style="Title.TLabel", anchor="w").grid(row=0, column=0, sticky="w")
        ttk.Label(titles, textvariable=self.proc_title_var,
                  style="Title.TLabel", anchor="e").grid(row=0, column=1, sticky="e")

        # ===== Canvases =====
        canv_frame = ttk.Frame(self)
        canv_frame.grid(row=3, column=0, sticky="nsew", padx=12, pady=(0, 10))
        canv_frame.columnconfigure(0, weight=1)
        canv_frame.columnconfigure(1, weight=1)
        canv_frame.rowconfigure(0, weight=1)

        self.canvas_orig = tk.Canvas(
            canv_frame, bg=self.colors["canvas"],
            highlightthickness=1, highlightbackground=self.colors["border"]
        )
        self.canvas_proc = tk.Canvas(
            canv_frame, bg=self.colors["canvas"],
            highlightthickness=1, highlightbackground=self.colors["border"]
        )
        self.canvas_orig.grid(row=0, column=0, sticky="nsew", padx=(0, 16))
        self.canvas_proc.grid(row=0, column=1, sticky="nsew", padx=(16, 0))
        self._canvases = [self.canvas_orig, self.canvas_proc]

        # ===== Status =====
        self.status = ttk.Label(self, text="Open an image to begin.",
                                anchor="w", style="Subtle.TLabel")
        self.status.grid(row=4, column=0, sticky="ew", padx=12, pady=(0, 10))

        def _initial_render(_e=None): self._render_images()
        self.canvas_orig.bind("<Configure>", _initial_render)
        self.canvas_proc.bind("<Configure>", _initial_render)

    # helpers
    def _brighten(self, hex_color):
        c = hex_color.lstrip("#")
        r, g, b = int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)
        r = min(255, int(r * 1.12)); g = min(255, int(g * 1.12)); b = min(255, int(b * 1.12))
        return f"#{r:02x}{g:02x}{b:02x}"

    # ---------- Developer info ----------
    def _load_developer_info(self):
        info_path = "developer_info.json"
        name = "Nishat Tasnim Chowdhury"
        sid = "ID: 0812220105101022"
        photo = None
        if os.path.exists(info_path):
            try:
                with open(info_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    name = data.get("name", name)
                    sid = data.get("id", sid)
                    photo = data.get("photo", None)
            except Exception:
                pass

        # Prefer the packaged photo when no photo specified in JSON
        packaged_photo = resource_path("nishat.jpg")
        if (not photo) and os.path.exists(packaged_photo):
            photo = packaged_photo

        self.dev_name_var.set(name)
        self.dev_id_var.set(sid)
        self._set_dev_photo(photo)

    def _set_dev_photo(self, path: Optional[str]):
        try:
            if path and os.path.exists(path):
                im = Image.open(path).convert("RGB")
            else:
                im = Image.new("RGB", (96, 96), (30, 40, 48))
            im = im.resize((72, 72), Image.LANCZOS)   # slightly larger avatar
            self._dev_photo_tk = ImageTk.PhotoImage(im)
            self.dev_photo_label.configure(image=self._dev_photo_tk)
        except Exception:
            pass

    def _ask_developer_info(self):
        win = tk.Toplevel(self)
        win.title("Developer Info")
        win.resizable(False, False)
        ttk.Label(win, text="Full Name").grid(row=0, column=0, sticky="w", padx=8, pady=(8, 0))
        name_var = tk.StringVar(value=self.dev_name_var.get())
        ttk.Entry(win, textvariable=name_var, width=32, style="Entry.TEntry").grid(row=0, column=1, padx=8, pady=(8, 0))

        ttk.Label(win, text="Full ID").grid(row=1, column=0, sticky="w", padx=8, pady=(6, 0))
        id_var = tk.StringVar(value=self.dev_id_var.get())
        ttk.Entry(win, textvariable=id_var, width=32, style="Entry.TEntry").grid(row=1, column=1, padx=8, pady=(6, 0))

        ttk.Label(win, text="Photo").grid(row=2, column=0, sticky="w", padx=8, pady=(6, 0))
        photo_var = tk.StringVar(value="")
        row = ttk.Frame(win); row.grid(row=2, column=1, padx=8, pady=(6, 0), sticky="ew")
        ttk.Entry(row, textvariable=photo_var, width=24, style="Entry.TEntry").pack(side="left", padx=(0, 6))
        ttk.Button(row, text="Browse", command=lambda: photo_var.set(filedialog.askopenfilename(filetypes=[("Image", "*.jpg *.jpeg *.png")]))).pack(side="left")

        def save():
            self.dev_name_var.set(name_var.get())
            self.dev_id_var.set(id_var.get())
            self._set_dev_photo(photo_var.get() or None)
            try:
                with open("developer_info.json", "w", encoding="utf-8") as f:
                    json.dump({"name": self.dev_name_var.get(),
                               "id": self.dev_id_var.get(),
                               "photo": photo_var.get()}, f, indent=2)
            except Exception:
                pass
            win.destroy()

        ttk.Button(win, text="Save", command=save).grid(row=3, column=0, columnspan=2, pady=10)

    # ---------- File ops ----------
    def open_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff")])
        if not path:
            return
        try:
            im = Image.open(path)
        except Exception as e:
            messagebox.showerror("Open Image", f"Could not open:\n{e}")
            return

        self.img_open = im.convert("RGB")
        self.img = None
        self.open_filename = os.path.basename(path)
        self.undo_stack.clear(); self.redo_stack.clear()
        self._render_images()
        self.orig_title_var.set("Original Image")
        self.proc_title_var.set("Output Image — (apply any operation)")
        self._set_status(f"Loaded: {self.open_filename}  |  {self.img_open.width}×{self.img_open.height}")

    def save_as(self):
        if self.img is None:
            messagebox.showinfo("Save Output Image", "No result yet. Apply an operation first.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".png",
                    filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg *.jpeg"), ("BMP", "*.bmp"), ("TIFF", "*.tif *.tiff")])
        if not path:
            return
        try:
            self.img.save(path)
            self.proc_title_var.set("Output Image — saved")
            self._set_status(f"Saved to {os.path.basename(path)}")
        except Exception as e:
            messagebox.showerror("Save Output Image", f"Could not save:\n{e}")

    def reset_image(self):
        if self.img_open is None:
            return
        self.img = None
        self.undo_stack.clear(); self.redo_stack.clear()
        self._render_images()
        self.proc_title_var.set("Output Image — (apply any operation)")
        self._set_status("Reset (showing original only)")

    def undo(self):
        if not self.undo_stack:
            return
        self.redo_stack.append(self.img.copy() if self.img is not None else None)
        self.img = self.undo_stack.pop()
        self._render_images()
        self._set_status("Undo")

    def redo(self):
        if not self.redo_stack:
            return
        self.undo_stack.append(self.img.copy() if self.img is not None else None)
        self.img = self.redo_stack.pop()
        self._render_images()
        self._set_status("Redo")

    # ---------- Rendering ----------
    def _fit_to_canvas(self, im: Image.Image, canvas: tk.Canvas) -> ImageTk.PhotoImage:
        cw = int(canvas.winfo_width() or 10)
        ch = int(canvas.winfo_height() or 10)
        if cw < 20 or ch < 20:
            cw, ch = 400, 300
        w, h = im.width, im.height
        scale = min(cw / w, ch / h)
        disp = im.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
        return ImageTk.PhotoImage(disp)

    def _render_images(self):
        self.canvas_orig.delete("all")
        self.canvas_proc.delete("all")
        if self.img_open:
            tki1 = self._fit_to_canvas(self.img_open, self.canvas_orig); self._tki1 = tki1
            self.canvas_orig.create_image(self.canvas_orig.winfo_width() // 2,
                                          self.canvas_orig.winfo_height() // 2, image=tki1)
        if self.img:
            tki2 = self._fit_to_canvas(self.img, self.canvas_proc); self._tki2 = tki2
            self.canvas_proc.create_image(self.canvas_proc.winfo_width() // 2,
                                          self.canvas_proc.winfo_height() // 2, image=tki2)

    def _set_status(self, msg: str):
        self.status.configure(text=msg)

    # ---------- Apply helpers ----------
    def _apply_simple(self, func, name):
        if self.img_open is None:
            return
        src = self.img if self.img is not None else self.img_open
        arr = pil_to_array(src)
        t0 = time.time()
        out = func(arr)
        dt = (time.time() - t0) * 1000

        self.undo_stack.append(self.img.copy() if self.img is not None else None)
        self.redo_stack.clear()
        self.img = array_to_pil(out)
        self._render_images()
        self.proc_title_var.set(f"Output Image — {name}")
        self._set_status(f"{name} in {dt:.1f} ms | {self.img.width}×{self.img.height}")

    # ---------- small helpers for dialogs ----------
    def _odd_spinbox(self, parent, from_=3, to=11, initial=3):
        var = tk.IntVar(value=initial)
        def make_odd(*_):
            v = int(var.get())
            if v % 2 == 0:
                var.set(v + 1 if v < to else v - 1)
        sp = ttk.Spinbox(parent, from_=from_, to=to, increment=2,
                         textvariable=var, width=6, command=make_odd)
        sp.bind("<FocusOut>", make_odd)
        return sp, var

    def _labeled_scale(self, parent, text, from_, to, init, resolution=0.1, style="Blue.Horizontal.TScale"):
        row = ttk.Frame(parent); row.pack(fill="x", padx=10, pady=(8, 0))
        ttk.Label(row, text=text, width=18).pack(side="left")
        val = tk.DoubleVar(value=init)
        s = ttk.Scale(row, from_=from_, to=to, orient="horizontal",
                      variable=val, length=260, style=style)
        s.pack(side="left", padx=(6, 8))
        readout = ttk.Label(row, text=f"{init:.2f}")
        readout.pack(side="left")
        def on_move(*_):
            v = round(float(val.get()) / resolution) * resolution
            val.set(v)
            readout.configure(text=f"{v:.2f}")
        s.configure(command=lambda _=None: on_move())
        return val

    # ---------- Parameter dialogs ----------
    def _dlg_smoothing(self):
        if self.img_open is None: return
        win = tk.Toplevel(self); win.title("Smoothing")
        win.configure(bg=self.colors["bg"]); win.resizable(False, False)

        pad = 10
        row = ttk.Frame(win); row.pack(fill="x", padx=pad, pady=(pad, 0))
        ttk.Label(row, text="Type", width=18).pack(side="left")
        type_var = tk.StringVar(value="Box")
        ttk.Combobox(row, textvariable=type_var, values=["Box", "Gaussian"],
                     state="readonly", width=14, style="Cmb.TCombobox").pack(side="left", padx=(6, 0))

        row2 = ttk.Frame(win); row2.pack(fill="x", padx=pad, pady=(10, 0))
        ttk.Label(row2, text="Kernel size (odd)", width=18).pack(side="left")
        k_spin, k_var = self._odd_spinbox(row2, 3, 11, 3)
        k_spin.pack(side="left", padx=(6, 0))

        sigma_var = self._labeled_scale(win, "Sigma (Gaussian)", 0.5, 5.0, 1.0, resolution=0.1)

        def apply():
            src = self.img if self.img is not None else self.img_open
            arr = pil_to_array(src)
            size = int(k_var.get())
            t0 = time.time()
            if type_var.get() == "Gaussian":
                out = smooth_gaussian(arr, size=size, sigma=float(sigma_var.get()))
                name = f"Smoothing (Gaussian, k={size}, σ={float(sigma_var.get()):.2f})"
            else:
                out = smooth_box(arr, size=size)
                name = f"Smoothing (Box, k={size})"
            dt = (time.time() - t0) * 1000
            self.undo_stack.append(self.img.copy() if self.img is not None else None)
            self.redo_stack.clear()
            self.img = array_to_pil(out)
            self._render_images()
            self.proc_title_var.set(f"Output Image — {name}")
            self._set_status(f"{name} in {dt:.1f} ms")
            win.destroy()

        ttk.Button(win, text="Apply", command=apply).pack(pady=12)

    def _dlg_sharpen(self):
        if self.img_open is None: return
        win = tk.Toplevel(self); win.title("Sharpen (Unsharp Mask)")
        win.configure(bg=self.colors["bg"]); win.resizable(False, False)

        pad = 10
        row = ttk.Frame(win); row.pack(fill="x", padx=pad, pady=(pad, 0))
        ttk.Label(row, text="Kernel size", width=18).pack(side="left")
        k_spin, k_var = self._odd_spinbox(row, 3, 11, 5)
        k_spin.pack(side="left", padx=(6, 0))

        sigma_var = self._labeled_scale(win, "Sigma", 0.5, 5.0, 1.0, resolution=0.1)
        amount_var = self._labeled_scale(win, "Amount", 0.2, 3.0, 1.0, resolution=0.1)

        def apply():
            src = self.img if self.img is not None else self.img_open
            arr = pil_to_array(src)
            size = int(k_var.get())
            t0 = time.time()
            out = unsharp_mask(arr, size=size, sigma=float(sigma_var.get()), amount=float(amount_var.get()))
            dt = (time.time() - t0) * 1000
            self.undo_stack.append(self.img.copy() if self.img is not None else None)
            self.redo_stack.clear()
            self.img = array_to_pil(out)
            self._render_images()
            self.proc_title_var.set(f"Output Image — Sharpen (k={size}, σ={float(sigma_var.get()):.2f}, amt={float(amount_var.get()):.2f})")
            self._set_status(f"Sharpen in {dt:.1f} ms")
            win.destroy()

        ttk.Button(win, text="Apply", command=apply).pack(pady=12)

    def _dlg_resize(self):
        if self.img_open is None: return
        win = tk.Toplevel(self); win.title("Resize")
        win.configure(bg=self.colors["bg"]); win.resizable(False, False)

        pad = 10
        row = ttk.Frame(win); row.pack(fill="x", padx=pad, pady=(pad, 0))
        ttk.Label(row, text="Scale (%)", width=18).pack(side="left")
        scale_val = tk.IntVar(value=150)
        val_lbl = ttk.Label(row, text="150%"); val_lbl.pack(side="right")
        sc = ttk.Scale(row, from_=10, to=500, orient="horizontal",
                       variable=scale_val, length=260, style="Blue.Horizontal.TScale")
        sc.pack(side="left", padx=(6, 8))
        def s_upd(_=None):
            v = int(scale_val.get()); scale_val.set(v); val_lbl.configure(text=f"{v}%")
        sc.configure(command=s_upd); s_upd()

        row2 = ttk.Frame(win); row2.pack(fill="x", padx=pad, pady=(10, 0))
        ttk.Label(row2, text="Method", width=18).pack(side="left")
        method_var = tk.StringVar(value="Bilinear")
        ttk.Combobox(row2, textvariable=method_var, values=["Nearest", "Bilinear"],
                     state="readonly", width=12, style="Cmb.TCombobox").pack(side="left", padx=(6, 0))

        def apply():
            src = self.img if self.img is not None else self.img_open
            arr = pil_to_array(src)
            scale = max(1, int(scale_val.get()))
            new_w = max(1, int(arr.shape[1] * scale / 100.0))
            new_h = max(1, int(arr.shape[0] * scale / 100.0))
            t0 = time.time()
            if method_var.get() == "Nearest":
                out = resize_nearest(arr, new_h, new_w); mname = "Nearest"
            else:
                out = resize_bilinear(arr, new_h, new_w); mname = "Bilinear"
            dt = (time.time() - t0) * 1000
            self.undo_stack.append(self.img.copy() if self.img is not None else None)
            self.redo_stack.clear()
            self.img = array_to_pil(out)
            self._render_images()
            self.proc_title_var.set(f"Output Image — Resize {scale}% ({mname})")
            self._set_status(f"Resize in {dt:.1f} ms")
            win.destroy()

        ttk.Button(win, text="Apply", command=apply).pack(pady=12)

    def _dlg_threshold(self):
        if self.img_open is None: return
        win = tk.Toplevel(self); win.title("Thresholding")
        win.configure(bg=self.colors["bg"]); win.resizable(False, False)

        pad = 10
        row = ttk.Frame(win); row.pack(fill="x", padx=pad, pady=(pad, 0))
        ttk.Label(row, text="Manual T (0..255)", width=18).pack(side="left")
        t_val = tk.IntVar(value=128)
        val_lbl = ttk.Label(row, text=str(t_val.get())); val_lbl.pack(side="right")
        sc = ttk.Scale(row, from_=0, to=255, orient="horizontal",
                       variable=t_val, length=260, style="Blue.Horizontal.TScale")
        sc.pack(side="left", padx=(6, 8))
        def upd(_=None):
            v = int(round(float(t_val.get())))
            t_val.set(v); val_lbl.configure(text=str(v))
        sc.configure(command=upd); upd()

        btnrow = ttk.Frame(win); btnrow.pack(fill="x", padx=pad, pady=(12, 10))
        ttk.Button(btnrow, text="Apply Manual",
                   command=lambda: self._thresh_manual_close(win, t_val.get())
                   ).pack(side="left", expand=True, fill="x", padx=(0, 6))
        ttk.Button(btnrow, text="Otsu Auto",
                   command=lambda: self._thresh_otsu_close(win)
                   ).pack(side="left", expand=True, fill="x", padx=(6, 0))

    def _thresh_manual_close(self, win, T):
        src = self.img if self.img is not None else self.img_open
        arr = pil_to_array(src)
        t0 = time.time()
        out = threshold_apply(arr, int(T))
        dt = (time.time() - t0) * 1000
        self.undo_stack.append(self.img.copy() if self.img is not None else None)
        self.redo_stack.clear()
        self.img = array_to_pil(out)
        self._render_images()
        self.proc_title_var.set(f"Output Image — Threshold (T={int(T)})")
        self._set_status(f"Threshold in {dt:.1f} ms")
        win.destroy()

    def _thresh_otsu_close(self, win):
        src = self.img if self.img is not None else self.img_open
        arr = pil_to_array(src)
        t0 = time.time()
        T = otsu_threshold(arr)
        out = threshold_apply(arr, T)
        dt = (time.time() - t0) * 1000
        self.undo_stack.append(self.img.copy() if self.img is not None else None)
        self.redo_stack.clear()
        self.img = array_to_pil(out)
        self._render_images()
        self.proc_title_var.set(f"Output Image — Otsu (T={T})")
        self._set_status(f"Otsu in {dt:.1f} ms")
        win.destroy()

    def _dlg_gamma(self):
        if self.img_open is None: return
        win = tk.Toplevel(self); win.title("Gamma")
        win.configure(bg=self.colors["bg"]); win.resizable(False, False)

        gamma_var = self._labeled_scale(win, "Gamma (0.1–5.0)", 0.1, 5.0, 0.7, resolution=0.05)

        def apply():
            src = self.img if self.img is not None else self.img_open
            arr = pil_to_array(src)
            gamma = float(gamma_var.get())
            t0 = time.time()
            out = gamma_transform(arr, gamma)
            dt = (time.time() - t0) * 1000
            self.undo_stack.append(self.img.copy() if self.img is not None else None)
            self.redo_stack.clear()
            self.img = array_to_pil(out)
            self._render_images()
            self.proc_title_var.set(f"Output Image — Gamma (γ={gamma:.2f})")
            self._set_status(f"Gamma in {dt:.1f} ms")
            win.destroy()

        ttk.Button(win, text="Apply", command=apply).pack(pady=12)

    # ---------- Histogram Window ----------
    def show_histogram(self):
        src = self.img if self.img is not None else self.img_open
        if src is None: return
        hist = histogram_gray(pil_to_array(src))  # length 256
        win = tk.Toplevel(self); win.title("Histogram (Grayscale)")
        win.configure(bg=self.colors["bg"])
        W, H = 720, 320
        canvas = tk.Canvas(win, width=W, height=H, bg=self.colors["canvas"],
                           highlightthickness=0)
        canvas.pack(fill="both", expand=True, padx=10, pady=10)

        padL, padR, padT, padB = 40, 20, 20, 40
        maxh = int(hist.max()) if hist.max() > 0 else 1

        # Axes
        x0, y0 = padL, H - padB
        x1, y1 = W - padR, padT
        canvas.create_line(x0, y0, x1, y0, fill=self.colors["subtle"])  # x-axis
        canvas.create_line(x0, y0, x0, y1, fill=self.colors["subtle"])  # y-axis

        # X ticks 0..255 every 64
        for t in range(0, 256, 64):
            tx = x0 + int(t * (x1 - x0) / 255.0)
            canvas.create_line(tx, y0, tx, y0 + 6, fill=self.colors["subtle"])
            canvas.create_text(tx, y0 + 16, text=str(t), fill=self.colors["subtle"], font=("Segoe UI", 8))

        # Y ticks 0..max
        for i in range(5):
            frac = i / 4.0
            ty = y0 - int(frac * (y0 - y1))
            canvas.create_line(x0 - 6, ty, x0, ty, fill=self.colors["subtle"])
            canvas.create_text(x0 - 10, ty, text=f"{int(maxh*frac)}", fill=self.colors["subtle"],
                               font=("Segoe UI", 8), anchor="e")

        # Bars
        for i in range(256):
            xa = x0 + int(i * (x1 - x0) / 256.0)
            xb = x0 + int((i + 1) * (x1 - x0) / 256.0)
            h = int((hist[i] / maxh) * (y0 - y1))
            canvas.create_rectangle(xa, y0 - h, xb, y0,
                                    fill=self.colors["blue"], outline="")

if __name__ == "__main__":
    app = ImageToolkitApp()
    app.mainloop()
