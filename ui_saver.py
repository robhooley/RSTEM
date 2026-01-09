import os
import json
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple
from importlib.util import find_spec
from datetime import datetime
import shutil
import tempfile
from tqdm import tqdm
from time import sleep
import matplotlib.pyplot as plt

import cv2
import numpy as np


# --- imports that work both inside/outside the package ---
if find_spec("RSTEM.app_context") is not None:
    from RSTEM.app_context import get_app
    from RSTEM.acquisitions import (acquire_STEM,
                                    point_acquisition)
    from RSTEM.utilities import normalise_to_8bit
else:
    from app_context import get_app
    from acquisitions import (acquire_STEM,
                              point_acquisition)

    from utilities import normalise_to_8bit

from expertpi.api import DetectorType as DT

# -----------------------------
# State
# -----------------------------
@dataclass
class AppState:
    # Save directory chosen once in UI
    save_dir: Optional[str] = None

    # STEM
    stem_images: Optional[List[np.ndarray]] = None         # [img] or [img, img], uint16
    stem_labels: Optional[List[str]] = None                # ["BF"] / ["HAADF"] / ["BF","HAADF"]
    stem_metadata: Optional[Dict[str, Any]] = None

    # Camera
    camera_image: Optional[np.ndarray] = None
    camera_metadata: Optional[Dict[str, Any]] = None
    camera_metadata_path: Optional[str] = None             # metadata file produced by acquire_camera

import tkinter as tk
from typing import List, Tuple, Optional
import numpy as np


def tk_ginput(
    parent: tk.Misc,
    image: np.ndarray,
    n: int = -1,
    timeout: float = 0,
    title: str = "Click to select points (Right-click to remove / Enter to finish)",
    max_display_size: int = 1024,
    disp_radius: float = 2,
) -> List[Tuple[float, float]]:
    """
    Tkinter alternative to plt.ginput.

    Parameters
    ----------
    parent : tk.Misc
        Your Tk root or any widget (e.g. `self` inside your App class).
    image : np.ndarray
        2D grayscale or 3D HxWxC image. uint8/uint16 preferred.
        Display is 8-bit (uint16 is scaled for viewing).
    n : int
        Number of points to collect. -1 means unlimited (until finish).
    timeout : float
        Seconds before auto-finish. 0 means no timeout.
    title : str
        Window title/instructions.
    max_display_size : int
        Max size (pixels) for the display window; image is downscaled for viewing.

    Returns
    -------
    points : list of (x, y)
        Pixel coordinates in the *original image coordinate system* (float).
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a numpy array")

    img = image
    if img.ndim == 3 and img.shape[2] in (3, 4):
        # convert RGB/RGBA to grayscale for display if you want;
        # or keep RGB for display (Tk PPM supports RGB)
        pass

    # --- Prepare an 8-bit display image ---
    disp = img
    if disp.ndim == 3 and disp.shape[2] == 4:
        disp = disp[:, :, :3]

    if disp.ndim == 2:
        if disp.dtype == np.uint16:
            # fast view scaling for uint16 (good enough for clicking)
            d = (disp >> 8).astype(np.uint8)
        elif disp.dtype == np.uint8:
            d = disp
        else:
            # normalize floats/other ints to 0..255
            d = disp.astype(np.float32)
            d -= float(np.min(d))
            mx = float(np.max(d))
            if mx > 0:
                d = d / mx
            d = (d * 255).astype(np.uint8)
        disp8 = d
        mode = "L"
    elif disp.ndim == 3 and disp.shape[2] == 3:
        if disp.dtype == np.uint16:
            disp8 = (disp >> 8).astype(np.uint8)
        elif disp.dtype == np.uint8:
            disp8 = disp
        else:
            d = disp.astype(np.float32)
            d -= float(np.min(d))
            mx = float(np.max(d))
            if mx > 0:
                d = d / mx
            disp8 = (d * 255).astype(np.uint8)
        mode = "RGB"
    else:
        raise ValueError(f"Unsupported image shape: {img.shape}")

    H, W = img.shape[:2]

    # --- Downscale for display if too large (nearest-neighbor) ---
    scale = 1.0
    dispH, dispW = disp8.shape[:2]
    max_dim = max(dispH, dispW)
    if max_dim > max_display_size:
        scale = max_display_size / max_dim
        newW = max(1, int(dispW * scale))
        newH = max(1, int(dispH * scale))
        # nearest-neighbor downscale
        yy = (np.linspace(0, dispH - 1, newH)).astype(np.int32)
        xx = (np.linspace(0, dispW - 1, newW)).astype(np.int32)
        if mode == "L":
            disp8 = disp8[yy][:, xx]
        else:
            disp8 = disp8[yy][:, xx, :]
        dispH, dispW = disp8.shape[:2]

    # --- Build a Toplevel window and Canvas ---
    top = tk.Toplevel(parent)
    top.title(title)
    top.transient(parent.winfo_toplevel())
    top.grab_set()  # modal
    top.focus_force()

    points: List[Tuple[float, float]] = []
    marker_ids: List[int] = []
    finished = {"done": False}

    # Create PPM data for Tk PhotoImage (no PIL required)
    if mode == "L":
        rgb = np.repeat(disp8[:, :, None], 3, axis=2)
    else:
        rgb = disp8

    # PPM (P6) header + binary pixels
    header = f"P6\n{dispW} {dispH}\n255\n".encode("ascii")
    ppm = header + rgb.tobytes()

    photo = tk.PhotoImage(data=ppm, format="PPM")

    canvas = tk.Canvas(top, width=dispW, height=dispH, highlightthickness=0)
    canvas.pack(padx=8, pady=8)
    canvas.create_image(0, 0, image=photo, anchor="nw")

    info = tk.StringVar(value="Left-click: add point | Right-click or Enter: finish | Backspace: undo | Esc: cancel")
    ttk_label = tk.Label(top, textvariable=info, anchor="w")
    ttk_label.pack(fill="x", padx=8, pady=(0, 8))

    def finish():
        finished["done"] = True

    def cancel():
        points.clear()
        finished["done"] = True

    def undo(_evt=None):
        if points:
            points.pop()
        if marker_ids:
            canvas.delete(marker_ids.pop())
        info.set(f"{len(points)} point(s)")

    def add_point(event):
        if finished["done"]:
            return
        if n != -1 and len(points) >= n:
            return

        # event coords are in display coords -> convert back to original image coords
        x_disp, y_disp = float(event.x), float(event.y)
        x_img = x_disp / scale
        y_img = y_disp / scale

        # clamp to bounds
        x_img = max(0.0, min(float(W - 1), x_img))
        y_img = max(0.0, min(float(H - 1), y_img))

        points.append((x_img, y_img))

        # draw marker in display coords
        r = disp_radius
        mid = canvas.create_oval(
            x_disp - r, y_disp - r, x_disp + r, y_disp + r,
            outline="yellow", width=2
        )
        marker_ids.append(mid)

        info.set(f"{len(points)} point(s)")
        if n != -1 and len(points) >= n:
            finish()

    def right_click(_event):
        finish()

    # Bindings
    canvas.bind("<Button-1>", add_point)  # left click → add point
    canvas.bind("<Button-3>", undo)  # right click → undo
    top.bind("<BackSpace>", undo)  # backspace → undo
    canvas.bind("<Button-2>", lambda _e: finish())  # middle click → finish
    top.bind("<Return>", lambda _e: finish())  # Enter → finish
    top.bind("<Escape>", lambda _e: cancel())  # Esc → cancel      # cancel

    # Optional timeout
    if timeout and timeout > 0:
        top.after(int(timeout * 1000), finish)

    # Wait until done (modal)
    while not finished["done"]:
        top.update_idletasks()
        top.update()

    top.grab_release()
    top.destroy()
    return points

# -----------------------------
# Acquisition hooks
# -----------------------------
def infer_stem_labels_from_insertion(num_images: int) -> List[str]:
    """
    Best-effort label inference that matches your acquire_STEM detector selection:
      - If both detectors inserted -> BF + HAADF
      - If only HAADF inserted -> HAADF
      - Otherwise -> BF (including off-axis BF)
    """
    if num_images == 2:
        return ["BF", "HAADF"]

    app = get_app()
    bf_in = app.api.stem_detector.get_is_inserted(DT.BF)
    haadf_in = app.api.stem_detector.get_is_inserted(DT.HAADF)

    if haadf_in and (not bf_in):
        return ["HAADF"]

    return ["BF"]


def acquire_stem_data() -> Tuple[List[np.ndarray], List[str], Dict[str, Any]]:
    """
    Calls your real acquisition.
    Must return (images_list, labels_list, metadata_dict).
    acquire_STEM assumed to return uint16 images.
    """
    images, metadata = acquire_STEM()  # expected: (list_of_images, metadata_dict)
    if not isinstance(images, (list, tuple)):
        raise TypeError("acquire_STEM() must return (list_of_images, metadata_dict)")

    images = list(images)
    labels = infer_stem_labels_from_insertion(len(images))

    if len(labels) != len(images):
        # fallback: generic labels
        labels = [f"CH{i}" for i in range(len(images))]

    return images, labels, metadata


def acquire_camera_data(self) -> Tuple[np.ndarray, Dict[str, Any], str]:
    """
    Replace with your real camera acquisition.
    Must return: (image_array, metadata_dict, metadata_file_path)
    Demo writes a metadata file.
    """
    app =get_app()

    """    bf_in = app.api.stem_detector.get_is_inserted(DT.BF)
    haadf_in = app.api.stem_detector.get_is_inserted(DT.HAADF)
    if bf_in or haadf_in:
        if bf_in:
            app.detectors.stem.insert_bf(False)
        if haadf_in:
            app.detectors.stem.insert_df(False)
        for _ in tqdm(range(5), desc="Stabilising after STEM detector retraction", unit=""):
            sleep(1)
    app.scanning.set_off_axis(False)
    sleep(0.2)  # stabilisation after deflector change"""

    overview_stem = ((np.random.rand(1024, 1024) * 255).astype(np.uint8),{"Beam diameter (d50) (nm)":100,"Pixel size (nm)":2})#acquire_STEM()
    #overview_stem = acquire_STEM()
    stem_metadata = overview_stem[1]
    beam_size = stem_metadata["Beam diameter (d50) (nm)"]
    pixel_size_nm = stem_metadata["Pixel size (nm)"]
    beam_size_pixels = beam_size/pixel_size_nm

    if beam_size_pixels <2:
        points = tk_ginput(self, overview_stem[0], n=-1, timeout=0)
    else:
        points = tk_ginput(self, overview_stem[0], n=-1, timeout=0,disp_radius=beam_size_pixels/2)

    imgs_list = []
    metadata_list = []
    for i in range(len(points)):
        if i == 0:
            img = ((np.random.rand(512, 512) * 255).astype(np.uint16),
                   {"Beam diameter (d50) (nm)": 100, "Pixel size (nm)": 2}) #return metadata=True
            #img = point_acquisition(points[i],return_metadata=True)
            imgs_list.append(img[0])
            metadata_list.append(img[1])
        else:
            #img = point_acquisition(points[i],return_metadata=False)
            img = (np.random.rand(512, 512) * 255).astype(np.uint16)
            imgs_list.append(img) #return_metadata=False

    meta = metadata_list[0]

    tmp_dir = tempfile.mkdtemp(prefix="camera_acq_")
    meta_path = os.path.join(tmp_dir, "camera_metadata.json")


    return imgs_list, meta, meta_path


# -----------------------------
# Save helpers
# -----------------------------
def save_stem_payload(
    stem_images: List[np.ndarray],
    stem_labels: List[str],
    metadata: Dict[str, Any],
    directory: str,
) -> None:
    """
    Saves 1–2 uint16 STEM images with detector-aware names + metadata json.
    Filenames: STEM_<timestamp>_<LABEL>.tiff
    """
    os.makedirs(directory, exist_ok=True)

    if not stem_images or not isinstance(stem_images, (list, tuple)):
        raise ValueError("stem_images must be a non-empty list/tuple of images")

    if not stem_labels or len(stem_labels) != len(stem_images):
        raise ValueError("stem_labels must be provided and match stem_images length")

    for i, img in enumerate(stem_images):
        if not isinstance(img, np.ndarray):
            raise TypeError(f"STEM image {i} is not a numpy array")
        if img.dtype != np.uint16:
            raise TypeError(f"STEM image {i} must be uint16, got {img.dtype}")
        if img.ndim != 2:
            raise ValueError(f"STEM image {i} must be 2D (H, W), got shape {img.shape}")

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base = f"STEM_{stamp}"

    for img, label in zip(stem_images, stem_labels):
        safe_label = "".join(c for c in str(label) if c.isalnum() or c in ("_", "-")).strip("_-") or "CH"
        path = os.path.join(directory, f"{base}_{safe_label}.tiff")
        ok = cv2.imwrite(path, img)  # uint16 TIFF supported
        if not ok:
            raise RuntimeError(f"cv2.imwrite failed for {path}")

    with open(os.path.join(directory, f"{base}_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def save_camera_payload(img_list: list, meta: Dict[str, Any], out_dir: str) -> None:
    """
    Saves camera image + metadata dict + copies the acquisition metadata file.
    """
    os.makedirs(out_dir, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base = f"CAMERA_{stamp}"
    print(meta)
    for i in range(len(img_list)):

        cv2.imwrite(os.path.join(out_dir, f"{base}_{i}.tiff"), img_list[i])

    with open(os.path.join(out_dir, f"{base}_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)



# -----------------------------
# UI
# -----------------------------
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("STEM / Camera Control")
        self.geometry("460x280")
        self.resizable(False, False)

        self.state = AppState()

        frame = ttk.Frame(self, padding=20)
        frame.pack(expand=True, fill="both")

        # Directory row
        ttk.Button(frame, text="Set Save Directory", command=self.on_set_directory).grid(
            row=0, column=0, columnspan=1, padx=6, pady=(0, 10), sticky="ew"
        )
        self.dir_var = tk.StringVar(value="(not set)")
        ttk.Label(frame, textvariable=self.dir_var).grid(
            row=0, column=1, columnspan=1, padx=6, pady=(0, 10), sticky="w"
        )

        # Buttons
        ttk.Button(frame, text="Acquire STEM", command=self.on_acquire_stem).grid(
            row=1, column=0, padx=6, pady=6, sticky="ew"
        )
        ttk.Button(frame, text="Save STEM", command=self.on_save_stem).grid(
            row=2, column=0, padx=6, pady=6, sticky="ew"
        )
        ttk.Button(frame, text="Acquire Camera", command=self.on_acquire_camera).grid(
            row=1, column=1, padx=6, pady=6, sticky="ew"
        )
        ttk.Button(frame, text="Save Camera", command=self.on_save_camera).grid(
            row=2, column=1, padx=6, pady=6, sticky="ew"
        )

        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=2)

        self.status_var = tk.StringVar(value="Idle")
        ttk.Label(frame, textvariable=self.status_var).grid(
            row=3, column=0, columnspan=2, pady=(15, 0), sticky="w"
        )

    def on_set_directory(self):
        out_dir = filedialog.askdirectory(title="Select default save folder")
        if not out_dir:
            return
        self.state.save_dir = out_dir
        self.dir_var.set(out_dir)

    def _require_save_dir(self) -> Optional[str]:
        """
        Returns the active save directory.
        If not set, prompts once and stores it.
        """
        if self.state.save_dir and os.path.isdir(self.state.save_dir):
            return self.state.save_dir

        out_dir = filedialog.askdirectory(title="Select default save folder")
        if not out_dir:
            return None

        self.state.save_dir = out_dir
        self.dir_var.set(out_dir)
        return out_dir

    def on_acquire_stem(self):
        try:
            self.status_var.set("Acquiring STEM...")
            self.update_idletasks()

            images, labels, meta = acquire_stem_data()

            self.state.stem_images = images
            self.state.stem_labels = labels
            self.state.stem_metadata = meta

            first = images[0]
            self.status_var.set(f"STEM acquired: {', '.join(labels)} | {first.shape} | {first.dtype}")
        except Exception as e:
            self.status_var.set("Idle")
            messagebox.showerror("Acquire STEM failed", str(e))

    def on_save_stem(self):
        if not self.state.stem_images or self.state.stem_metadata is None or not self.state.stem_labels:
            messagebox.showwarning("Nothing to save", "No STEM data available. Click 'Acquire STEM' first.")
            return

        out_dir = self._require_save_dir()
        if not out_dir:
            return

        try:
            self.status_var.set("Saving STEM...")
            self.update_idletasks()

            save_stem_payload(self.state.stem_images, self.state.stem_labels, self.state.stem_metadata, out_dir)
            del self.state.stem_images,self.state.stem_labels,self.state.stem_metadata #flush the image info after saving

            self.status_var.set("STEM saved.")
        except Exception as e:
            self.status_var.set("Idle")
            messagebox.showerror("Save STEM failed", str(e))

    def on_acquire_camera(self):
        try:
            self.status_var.set("Acquiring Camera...")
            self.update_idletasks()

            img, meta, meta_path = acquire_camera_data(self)

            self.state.camera_image = img
            self.state.camera_metadata = meta
            self.state.camera_metadata_path = meta_path

            self.status_var.set(f"Camera acquired: {img[0].shape}, {img[0].dtype}")
        except Exception as e:
            self.status_var.set("Idle")
            messagebox.showerror("Acquire Camera failed", str(e))

    def on_save_camera(self):
        img = self.state.camera_image
        meta = self.state.camera_metadata


        if img is None or meta is None:
            messagebox.showwarning("Nothing to save", "No camera data available. Click 'Acquire Camera' first.")
            return

        out_dir = self._require_save_dir()
        if not out_dir:
            return

        try:
            self.status_var.set("Saving Camera...")
            self.update_idletasks()

            save_camera_payload(img, meta, out_dir)
            del self.state.camera_image, self.state.camera_metadata  # flush the image info after saving
            self.status_var.set("Camera saved.")
        except Exception as e:
            self.status_var.set("Idle")
            messagebox.showerror("Save Camera failed", str(e))


def run_ui():
    app = App()
    app.mainloop()


if __name__ == "__main__":
    run_ui()
