import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox, ttk
from natsort import natsorted
import rsciio.blockfile as rs_blo
import rsciio.hspy      as rs_hspy
import numpy as np
from PIL import Image, ImageTk, ImageDraw
import cv2 as cv2
import os
from expert_pi.RSTEM.analysis_functions import *
from tqdm import tqdm
import fnmatch
import tifffile as tiff
#import multiprocessing as mp
import threading,queue

# Global variable to control mouse motion functionality
mouse_motion_enabled = True  # Initially enabled


def ask_scan_width(parent, guessed_width: int, num_files: int) -> int | None:
    """Modal integer prompt with initial value and proper parenting."""
    return simpledialog.askinteger(
        title="Enter Scan Width",
        prompt=f"Detected {num_files} images.\nGuessed width: {guessed_width}\nEnter scan width (X):",
        initialvalue=int(guessed_width),
        minvalue=1,
        maxvalue=num_files,
        parent=parent
    )
def ask_pixel_size_nm(parent, guessed_nm: float | None = None) -> float | None:
    return simpledialog.askfloat(
        title="Pixel size (nm/pixel)",
        prompt="Enter pixel size in nanometers (nm/pixel):",
        initialvalue=None if guessed_nm is None else float(guessed_nm),
        minvalue=0.0,
        parent=parent
    )

def _normalize_navigator(img2d: np.ndarray) -> np.ndarray:
    a = np.asarray(img2d, dtype=np.float32)
    mask = np.isfinite(a)
    vals = a[mask]
    if vals.size == 0:
        return np.zeros_like(a, dtype=np.uint8)
    lo, hi = np.percentile(vals, (1, 99))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(vals.min()), float(vals.max())
        if hi <= lo:
            return np.zeros_like(a, dtype=np.uint8)
    a = np.clip(a, lo, hi)
    den = (a.max() - a.min()) or 1.0
    return ((a - a.min()) / den * 255).astype(np.uint8)

def downsample_diffraction(array4d, rescale_to=128, mode='sum'):
    if array4d.ndim != 4:
        raise ValueError("Expected a 4-D array (scan_y, scan_x, dp_y, dp_x)")
    ny_s, nx_s, ny_dp, nx_dp = array4d.shape
    if ny_dp % rescale_to or nx_dp % rescale_to:
        raise ValueError(f"DP size must be a multiple of {rescale_to}; got ({ny_dp}, {nx_dp}).")
    fy, fx = ny_dp // rescale_to, nx_dp // rescale_to
    if fy != fx:
        raise ValueError("Diffraction pattern must be square.")
    a = np.ascontiguousarray(array4d)
    v = a.reshape(ny_s, nx_s, rescale_to, fy, rescale_to, fx)
    if mode == 'sum':
        return v.sum(axis=(-1, -3))
    elif mode == 'mean':
        return v.mean(axis=(-1, -3))
    raise ValueError("mode must be 'sum' or 'mean'")

def array2blo(data: np.ndarray, px_nm: float, save_path: str,
              intensity_scaling="crop", bin_data_to_128=True):
    """Save (Ny, Nx, Dy, Dx) array to .blo via rsciio with minimal metadata."""
    from rsciio.blockfile import file_writer

    # Optional binning to 128×128 DPs
    if bin_data_to_128 and (data.shape[2] != 128 or data.shape[3] != 128):
        data = downsample_diffraction(data, rescale_to=128, mode="sum")

    if data.ndim != 4:
        raise ValueError("`data` must be 4-D (Ny, Nx, Dy, Dx)")
    Ny, Nx, Dy, Dx = map(int, data.shape)

    # Navigator image (for preview in .blo viewers)
    navigator = _normalize_navigator(data.sum(axis=(2, 3)))

    # Minimal metadata
    meta = {"Pixel size (nm)": float(px_nm)}
    axes = [
        {"name": "scan_y", "units": "nm", "index_in_array": 0,
         "size": Ny, "offset": 0.0, "scale": px_nm, "navigate": True},
        {"name": "scan_x", "units": "nm", "index_in_array": 1,
         "size": Nx, "offset": 0.0, "scale": px_nm, "navigate": True},
        {"name": "qy", "units": "px", "index_in_array": 2,
         "size": Dy, "offset": 0.0, "scale": 1.0, "navigate": False},
        {"name": "qx", "units": "px", "index_in_array": 3,
         "size": Dx, "offset": 0.0, "scale": 1.0, "navigate": False},
    ]

    signal = {
        "data": data,
        "axes": axes,
        "metadata": {"acquisition": meta},
        "original_metadata": meta,
        "attributes": {"_lazy": False},
    }

    file_writer(
        save_path,
        signal,
        intensity_scaling=intensity_scaling,
        navigator=navigator,
        endianess="<",
        show_progressbar=True,
    )

def visualiser():
    """
    Creates a Tkinter application for visualizing a 4D numpy array.
    """
    root = tk.Tk()
    root.title("4D Array Visualizer")

    # Global variables to store the data and selected function
    data_array = None
    selected_function = tk.StringVar(value="BF")
    radius_value = tk.IntVar(value=50)  # Default radius value
    circle_center = [None, None]  # Default circle center (to be dynamically set)

    # Predefined functions for processing
    functions = {
        "BF": VBF,
        "DF": VDF,
        "ADF": VADF
    }

    # Canvas and state variables
    main_image_pil = None
    pointer_image_pil = None
    clicked_positions = []  # Store positions of clicks
    image_refs = {"main_image": None, "current_sub_image": None}  # Prevent garbage collection



    def save_images():
        """
        Saves the left image (main image with markers), sub-images for clicked positions,
        and the list of clicked positions to disk.
        """
        if data_array is None or main_image_pil is None:
            print("No data or main image available to save.")
            return

        # Ask user to select a directory
        save_dir = filedialog.askdirectory(parent=root)
        if not save_dir:
            print("Save canceled.")
            return

        # Save the main image with markers
        main_image_with_markers = main_image_pil.copy()
        draw = ImageDraw.Draw(main_image_with_markers)
        for i, (x, y) in enumerate(clicked_positions, start=1):
            # Draw markers on the main image
            draw.line((x - 3, y, x + 3, y), fill="red", width=1)
            draw.line((x, y - 3, x, y + 3), fill="red", width=1)
            draw.text((x + 5, y - 5), str(i), fill="red")

        # Save the main image
        main_image_path = f"{save_dir}/main_image_with_markers.tiff"
        main_image_with_markers.save(main_image_path)

        # Save individual sub-images
        for i, (x, y) in enumerate(clicked_positions, start=1):
            if 0 <= y < data_array.shape[0] and 0 <= x < data_array.shape[1]:
                sub_image = data_array[y, x]
                sub_image_path = f"{save_dir}/clicked_position_{i}_sub_image.tiff"
                Image.fromarray(sub_image).save(sub_image_path)

        # Save clicked positions to a text file
        positions_path = f"{save_dir}/clicked_positions.txt"
        with open(positions_path, "w") as f:
            for i, (x, y) in enumerate(clicked_positions, start=1):
                f.write(f"Position {i}: ({x}, {y})\n")

        print(f"Images and positions saved to {save_dir}")
        messagebox.showinfo("Save Complete", f"Images and positions saved to {save_dir}")


    def load_series():
        """Threaded TIFF series loader with progress bar + Cancel."""
        nonlocal data_array, circle_center

        folder_path = filedialog.askdirectory(mustexist=True, parent=root)
        if not folder_path:
            return

        files = [f for f in os.listdir(folder_path)
                 if os.path.isfile(os.path.join(folder_path, f))
                 and f.lower().endswith(('.tif', '.tiff'))]
        if not files:
            messagebox.showwarning("Load TIFF Series", "No .tif/.tiff files found.", parent=root)
            return

        try:
            files = natsorted(files)
        except Exception:
            files = sorted(files)
        N = len(files)

        guessed_scan_width = int(np.sqrt(N))
        scan_width = ask_scan_width(root, guessed_scan_width, N)
        if not scan_width:
            return
        if N % scan_width != 0:
            messagebox.showwarning("Load TIFF Series",
                                   f"{N} not divisible by {scan_width}; last row will be padded.",
                                   parent=root)
        scan_height = (N + (scan_width - 1)) // scan_width  # ceil

        # Progress UI
        progress_frame = tk.Frame(root)
        progress_frame.pack(side=tk.TOP, pady=10)
        progress_bar = ttk.Progressbar(progress_frame, orient="horizontal", length=320,
                                       mode="determinate", maximum=N)
        progress_bar.pack(side=tk.LEFT, padx=10)
        progress_label = tk.Label(progress_frame, text=f"0 / {N} (0.0%)")
        progress_label.pack(side=tk.LEFT, padx=5)
        cancel_flag = {'stop': False}

        def _cancel():
            cancel_flag['stop'] = True

        cancel_btn = tk.Button(progress_frame, text="Cancel", command=_cancel)
        cancel_btn.pack(side=tk.LEFT, padx=10)

        # Thread-safe queue for progress and result
        q = queue.Queue()

        def _read_one(p):
            try:
                import tifffile as _tiff
                return _tiff.imread(p)
            except Exception:
                from PIL import Image
                with Image.open(p) as im:
                    return np.array(im)

        def worker():
            try:
                first = _read_one(os.path.join(folder_path, files[0]))
                if first.ndim != 2:
                    q.put(('error', f"Expected 2D frames, got {first.shape}"))
                    return
                H, W = first.shape
                dtype = first.dtype
                stack = np.empty((N, H, W), dtype=dtype)
                stack[0] = first
                q.put(('progress', 1, N))

                for i, name in enumerate(files[1:], start=2):
                    if cancel_flag['stop']:
                        q.put(('cancelled', None, None))
                        return
                    arr = _read_one(os.path.join(folder_path, name))
                    if arr.shape != (H, W):
                        q.put(('error', f"Frame {name} shape {arr.shape} != {(H, W)}"))
                        return
                    stack[i - 1] = arr
                    q.put(('progress', i, N))

                # Pad if needed, then reshape to (Y, X, H, W)
                if N % scan_width != 0:
                    missing = scan_width - (N % scan_width)
                    pad = np.zeros((missing, H, W), dtype=dtype)
                    stack = np.concatenate([stack, pad], axis=0)
                data = stack.reshape(-1, scan_width, H, W)  # rows auto (ceil)
                q.put(('done', data, (H, W)))
            except Exception as e:
                q.put(('error', str(e)))

        def pump():
            try:
                while True:
                    kind, a, b = q.get_nowait()
                    if kind == 'progress':
                        i, total = a, b
                        progress_bar['value'] = i
                        progress_label.config(text=f"{i} / {total} ({(i / total) * 100:.1f}%)")
                    elif kind == 'done':
                        nonlocal data_array, circle_center
                        data_array = a
                        H, W = b
                        circle_center[0], circle_center[1] = W // 2, H // 2
                        update_pointer_image(data_array.shape[1] // 2, data_array.shape[0] // 2)
                        progress_frame.destroy()
                        messagebox.showinfo("Import Complete",
                                            f"Imported {N} images → navigator {data_array.shape[:2]}", parent=root)
                        update_function()
                        return
                    elif kind == 'cancelled':
                        progress_frame.destroy()
                        messagebox.showinfo("Load TIFF Series", "Cancelled.", parent=root)
                        return
                    elif kind == 'error':
                        progress_frame.destroy()
                        messagebox.showerror("Load TIFF Series", a, parent=root)
                        return
            except queue.Empty:
                pass
            root.after(33, pump)  # ~30 Hz UI updates

        threading.Thread(target=worker, daemon=True).start()
        pump()

    def resize_image(image, max_dimension):
        """
        Resizes an image to fit within the specified maximum dimension, preserving aspect ratio.
        Uses nearest-neighbor interpolation to keep the image pixelated.
        """
        original_width, original_height = image.size
        aspect_ratio = original_width / original_height

        if original_width > original_height:
            # Fit by width
            new_width = max_dimension
            new_height = int(new_width / aspect_ratio)
        else:
            # Fit by height
            new_height = max_dimension
            new_width = int(new_height / aspect_ratio)

        return image.resize((new_width, new_height), Image.NEAREST)

    def ensure_rgb(image):
        """
        Ensures the image is in RGB format. If grayscale, converts it to RGB.
        """
        if image.mode != "RGB":
            image = image.convert("RGB")
        return image

    def normalize_to_8bit(img, clip=(0.5, 99.5), ignore_zeros=True, use_log=False, apply_clahe=False):
        a = np.asarray(img).astype(np.float32)
        mask = np.isfinite(a)
        if ignore_zeros: mask &= (a != 0)
        if not mask.any(): return np.zeros(a.shape, np.uint8)
        vals = a[mask]
        lo, hi = np.percentile(vals, clip)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo: lo, hi = vals.min(), vals.max()
        if hi <= lo: v = lo; eps = max(abs(v) * 1e-3, 1e-6); lo, hi = v - eps, v + eps
        a = np.clip(a, lo, hi)
        if use_log:
            if lo <= 0: a += -(lo) + 1e-6
            a = np.log1p(a)
        denom = (a.max() - a.min()) or 1.0
        out = ((a - a.min()) / denom * 255).astype(np.uint8)
        if apply_clahe:
            try:
                import cv2;
                out = cv2.createCLAHE(2.0, (8, 8)).apply(out)
            except:  # fallback hist eq
                hist, _ = np.histogram(out, 256, (0, 255));
                cdf = hist.cumsum();
                cdf = (cdf - cdf[cdf > 0].min()) * 255 / (cdf.max() - cdf[cdf > 0].min());
                lut = np.clip(cdf, 0, 255).astype(np.uint8);
                out = lut[out]
        return out

    def update_main_image(trigger_user_function=False):
        """
        Updates the main image displayed in the left window.
        If `trigger_user_function` is True, regenerates the main image using the user-defined function.
        """
        nonlocal main_image_pil

        # Regenerate the main image if requested
        if trigger_user_function:
            if data_array is None:
                print("No data array available.")
                return
            function = functions[selected_function.get()]
            # Call the user-defined function to regenerate the main image
            new_main_image = function(data_array, radius_value.get(), tuple(circle_center))
            normalized_main_image = normalize_to_8bit(new_main_image)
            main_image_pil = Image.fromarray(normalized_main_image).convert("RGB")

        if main_image_pil is None:
            main_canvas.delete("all")
            return

        # Resize the main image to fit the left canvas
        resized_main = main_image_pil.resize((1024, 1024), Image.NEAREST)
        resized_main = ensure_rgb(resized_main)

        # Draw markers on the resized image
        draw = ImageDraw.Draw(resized_main)
        scale_factor_x = resized_main.width / main_image_pil.width
        scale_factor_y = resized_main.height / main_image_pil.height
        for i, (x, y) in enumerate(clicked_positions, start=1):
            # Scale marker positions to match the resized image
            scaled_x = int((x + 0.5) * scale_factor_x)
            scaled_y = int((y + 0.5) * scale_factor_y)
            draw.line((scaled_x - 3, scaled_y, scaled_x + 3, scaled_y), fill="red", width=1)
            draw.line((scaled_x, scaled_y - 3, scaled_x, scaled_y + 3), fill="red", width=1)
            draw.text((scaled_x + 5, scaled_y - 5), str(i), fill="red")

        # Update the main canvas
        main_image_tk = ImageTk.PhotoImage(resized_main)

        image_refs["main_image"] = main_image_tk
        main_canvas.create_image(0, 0, anchor=tk.NW, image=main_image_tk)

    def update_pointer_image(x, y):
        """
        Updates the right window's image based on the pointer position.
        Displays a semi-transparent circle with the radius set by the user.
        """
        nonlocal pointer_image_pil, circle_center
        if data_array is None or main_image_pil is None:
            return

        if 0 <= y < data_array.shape[0] and 0 <= x < data_array.shape[1]:
            sub_image = data_array[y, x]
            normalized = normalize_to_8bit(sub_image)
            pointer_image_pil = Image.fromarray(normalized).convert("RGB")

            # Resize the pointer image to fit the square canvas
            resized_pointer = pointer_image_pil.resize((SQUARE_CANVAS_SIZE, SQUARE_CANVAS_SIZE), Image.NEAREST)

            # Scale circle_center to the resized image dimensions
            scale_x = SQUARE_CANVAS_SIZE / pointer_image_pil.width
            scale_y = SQUARE_CANVAS_SIZE / pointer_image_pil.height
            upscaled_center_x = int(circle_center[0] * scale_x)
            upscaled_center_y = int(circle_center[1] * scale_y)

            # Draw the semi-transparent circle
            draw = ImageDraw.Draw(resized_pointer, "RGBA")
            circle_radius = radius_value.get()
            draw.ellipse(
                (upscaled_center_x - circle_radius, upscaled_center_y - circle_radius,
                 upscaled_center_x + circle_radius, upscaled_center_y + circle_radius),
                outline=(255, 0, 0, 128),  # Semi-transparent red
                width=3
            )
            draw.line((upscaled_center_x - 3, upscaled_center_y, upscaled_center_x + 3, upscaled_center_y), fill="red", width=1)
            draw.line((upscaled_center_x, upscaled_center_y - 3, upscaled_center_x, upscaled_center_y + 3), fill="red", width=1)

            pointer_image_tk = ImageTk.PhotoImage(resized_pointer)
            image_refs["current_sub_image"] = pointer_image_tk
            pointer_canvas.create_image(0, 0, anchor=tk.NW, image=pointer_image_tk)

    def on_pointer_click(event):
        """
        Handles clicks on the pointer image to set the circle's center.
        Maps the mouse click to the actual pointer image coordinates.
        """
        nonlocal circle_center, pointer_image_pil
        if pointer_image_pil is None:
            return

        # Get Canvas dimensions
        canvas_width = pointer_canvas.winfo_width()
        canvas_height = pointer_canvas.winfo_height()

        # Pointer image dimensions (square)
        image_width = pointer_image_pil.width
        image_height = pointer_image_pil.height

        # The canvas is forced to be square, so drawn size matches SQUARE_CANVAS_SIZE
        drawn_width = SQUARE_CANVAS_SIZE
        drawn_height = SQUARE_CANVAS_SIZE

        # Since the canvas is square, offsets are always 0
        x_offset = 0
        y_offset = 0

        # Debugging information
        #print(f"Canvas Size: {canvas_width}x{canvas_height}")
        #print(f"Image Size: {image_width}x{image_height}")
        #print(f"Drawn Size: {drawn_width}x{drawn_height}")
        #print(f"Offsets: x={x_offset}, y={y_offset}")
        #print(f"Click Position: x={event.x}, y={event.y}")

        # Scaling factors
        scale_x = image_width / drawn_width
        scale_y = image_height / drawn_height

        # Map the click position to the original image coordinates
        circle_center[0] = int((event.x - x_offset) * scale_x)
        circle_center[1] = int((event.y - y_offset) * scale_y)

        # Debugging output for circle center
        print(f"Circle Center Updated: {circle_center[0]}, {circle_center[1]}")

        # Update the pointer image to reflect the new circle center
        update_pointer_image(circle_center[0], circle_center[1])

        update_function()

    def on_click(event):
        """
        Handles left-click events on the main image to add markers.
        """
        #global clicked_positions
        if main_image_pil is None:
            return

        x = int(event.x * main_image_pil.width / main_canvas.winfo_width())
        y = int(event.y * main_image_pil.height / main_canvas.winfo_height())
        clicked_positions.append((x, y))
        update_main_image()

    def toggle_mouse_motion(event=None):
        """
        Toggles the mouse motion functionality on and off with right-click.
        """
        global mouse_motion_enabled  # Access the global variable
        mouse_motion_enabled = not mouse_motion_enabled
        state = "enabled" if mouse_motion_enabled else "suspended"
        print(f"Mouse motion functionality is now {state}.")

    def _on_mouse_motion(event):
        """
        Handles mouse motion over the main image.
        Updates the pointer image dynamically based on the cursor position.
        """
        global mouse_motion_enabled  # Access the global variable
        if not mouse_motion_enabled:  # Skip updates if mouse motion is suspended
            return

        if data_array is None:
            # If no data array is available, skip the update
            #print("No data array loaded.")
            return

        #update_pointer_image(x, y)
        # Map the mouse position to the data array
        y = int(event.y * data_array.shape[0] / main_canvas.winfo_height())
        x = int(event.x * data_array.shape[1] / main_canvas.winfo_width())
        # Ensure indices are within bounds
        y = max(0, min(data_array.shape[0] - 1, y))
        x = max(0, min(data_array.shape[1] - 1, x))

        update_pointer_image(x, y)

    def load_npy_with_progress(path):
        # This creates a memmap immediately
        mm = np.load(path, mmap_mode='r')
        out = np.empty_like(mm)
        total = mm.size
        chunk = max(1, total // 200)  # ~200 steps

        # UI
        frame = tk.Frame(root);
        frame.pack(side=tk.TOP, pady=10)
        bar = ttk.Progressbar(frame, orient="horizontal", length=300, mode="determinate");
        bar.pack(side=tk.LEFT, padx=10)
        lbl = tk.Label(frame, text="0%");
        lbl.pack(side=tk.LEFT, padx=5)

        done = 0
        flat_src = mm.ravel()
        flat_dst = out.ravel()
        for start in range(0, total, chunk):
            end = min(total, start + chunk)
            flat_dst[start:end] = flat_src[start:end]
            done = end
            bar["value"] = (done / total) * 100
            lbl.config(text=f"{bar['value']:.1f}%")
            root.update_idletasks()

        frame.destroy()
        return out

        """    def load_data():
        """
        #Loads a .npy file as a 4D data array.
        """
        nonlocal data_array
        file_path = filedialog.askopenfilename(parent=root,filetypes=[("NumPy Array", "*.npy")])
        if file_path:
            #data_array = np.load(file_path)
            #np.flip(np.flip(data_array, axis=0), axis=1)
            #update_function()
            data_array = load_npy_with_progress(file_path)#np.load(file_path)
            H, W = data_array.shape[2], data_array.shape[3]
            circle_center[0] = W // 2
            circle_center[1] = H // 2
            update_pointer_image(data_array.shape[1] // 2, data_array.shape[0] // 2)
            update_function()"""

    # UPDATED: generic threaded, cancellable progress dialog
    def _load_with_progress(start_fn, title: str):
        """
        start_fn(cancel_flag: dict, q: queue.Queue) should push:
            ('status', 'message')    optional, update status label
            ('done', ndarray)        on success
            ('error', 'message')     on failure
        """
        progress_win = tk.Toplevel(root)
        progress_win.title(title)
        progress_win.transient(root)
        progress_win.grab_set()  # modal dialog

        tk.Label(progress_win, text=title).pack(padx=12, pady=(12, 4))

        bar = ttk.Progressbar(progress_win, orient="horizontal", length=340, mode="indeterminate")
        bar.pack(padx=12, pady=8)
        bar.start(10)  # ms per step

        status_lbl = tk.Label(progress_win, text="Starting…")
        status_lbl.pack(padx=12, pady=(0, 10))

        btn_frame = tk.Frame(progress_win);
        btn_frame.pack(pady=(0, 12))
        cancel_flag = {'stop': False}

        def _cancel():
            cancel_flag['stop'] = True
            status_lbl.config(text="Cancelling…")

        tk.Button(btn_frame, text="Cancel", command=_cancel).pack()

        q = queue.Queue()

        def worker():
            try:
                start_fn(cancel_flag, q)
            except Exception as e:
                q.put(('error', f"{type(e).__name__}: {e}"))

        def pump():
            try:
                while True:
                    kind, payload = q.get_nowait()
                    if kind == 'status':
                        status_lbl.config(text=str(payload))
                    elif kind == 'done':
                        # If user cancelled while load was in-flight, drop the result.
                        if cancel_flag['stop']:
                            bar.stop();
                            progress_win.grab_release();
                            progress_win.destroy()
                            messagebox.showinfo(title, "Cancelled.", parent=root)
                            return
                        arr = payload
                        arr4d = _coerce_to_4d(arr, root, ask_scan_width)
                        if arr4d is None:
                            bar.stop();
                            progress_win.grab_release();
                            progress_win.destroy()
                            return
                        nonlocal data_array, circle_center
                        data_array = arr4d
                        H, W = data_array.shape[2], data_array.shape[3]
                        circle_center[:] = [W // 2, H // 2]
                        update_pointer_image(data_array.shape[1] // 2, data_array.shape[0] // 2)
                        bar.stop();
                        progress_win.grab_release();
                        progress_win.destroy()
                        messagebox.showinfo(title, f"Loaded → navigator {data_array.shape[:2]}", parent=root)
                        update_function()
                        return
                    elif kind == 'error':
                        bar.stop();
                        progress_win.grab_release();
                        progress_win.destroy()
                        messagebox.showerror(title, str(payload), parent=root)
                        return
            except queue.Empty:
                pass
            root.after(50, pump)  # ~20 Hz UI update

        threading.Thread(target=worker, daemon=True).start()
        pump()

    def _coerce_to_4d(arr, parent, ask_width_fn):
        arr = np.asarray(arr)
        if arr.ndim == 4:
            return arr
        if arr.ndim == 3:
            N, H, W = arr.shape
            guessed = int(np.sqrt(N))
            scan_width = ask_width_fn(parent, guessed, N)
            if not scan_width:
                return None
            if N % scan_width != 0:
                missing = scan_width - (N % scan_width)
                arr = np.concatenate([arr, np.zeros((missing, H, W), dtype=arr.dtype)], axis=0)
                N = arr.shape[0]
            return arr.reshape(N // scan_width, scan_width, H, W)
        messagebox.showerror("File Load", f"Expected 3D or 4D array, got {arr.shape}", parent=parent)
        return None

    def _extract_rsciio_array(obj):
        """
        Robustly extract a NumPy array from rsciio reader output.
        Handles dicts with 'data', lists/tuples of such dicts, or arrays directly.
        Returns (np.ndarray, debug_info_str) or raises ValueError.
        """
        import numpy as _np

        def _from_dict(d):
            if not isinstance(d, dict):
                return None
            if "data" in d:
                arr = d["data"]
                if isinstance(arr, _np.ndarray):
                    return arr
            # common alternates some readers use
            for key in ("signals", "Signal", "items", "datasets"):
                if key in d:
                    v = d[key]
                    # try first element or iterate
                    if isinstance(v, (list, tuple)) and v:
                        for item in v:
                            got = _from_dict(item)
                            if got is not None:
                                return got
            return None

        # direct ndarray
        if isinstance(obj, _np.ndarray):
            return obj, f"type=ndarray shape={obj.shape} dtype={obj.dtype}"

        # dict top-level
        if isinstance(obj, dict):
            arr = _from_dict(obj)
            if arr is not None:
                return arr, f"type=dict→ndarray shape={arr.shape} dtype={arr.dtype} keys={list(obj.keys())}"
            raise ValueError(f"Dict had no usable 'data' (keys={list(obj.keys())})")

        # list/tuple top-level
        if isinstance(obj, (list, tuple)):
            dbg = []
            for i, item in enumerate(obj):
                if isinstance(item, _np.ndarray):
                    return item, f"type=list[{i}] ndarray shape={item.shape} dtype={item.dtype}"
                if isinstance(item, dict):
                    arr = _from_dict(item)
                    if arr is not None:
                        return arr, f"type=list[{i}] dict→ndarray shape={arr.shape} dtype={arr.dtype}"
                dbg.append(type(item).__name__)
            raise ValueError(f"List/tuple contained no ndarray/dict-with-data (inner types={dbg})")

        raise ValueError(f"Unsupported rsciio return type: {type(obj).__name__}")

    # UPDATED: cancellable .blo loader using rsciio
    def load_blo():
        nonlocal data_array, circle_center
        path = filedialog.askopenfilename(parent=root, filetypes=[("Blockfile", "*.blo"), ("All files", "*.*")])
        if not path:
            return

        def start_fn(cancel_flag, q):
            q.put(('status', "Reading .blo…"))
            if cancel_flag['stop']: return
            obj = rs_blo.file_reader(path)  # may be dict, list, or ndarray
            if cancel_flag['stop']: return
            try:
                arr, dbg = _extract_rsciio_array(obj)
            except Exception as e:
                q.put(('error', f"Extract failed: {e}"));
                return
            q.put(('status', f"Parsed: {dbg}"))
            q.put(('done', arr))

        _load_with_progress(start_fn, "Load .blo")

    # UPDATED: cancellable .hspy loader using rsciio
    def load_hspy():
        nonlocal data_array, circle_center
        path = filedialog.askopenfilename(parent=root, filetypes=[("HyperSpy", "*.hspy"), ("All files", "*.*")])
        if not path:
            return

        def start_fn(cancel_flag, q):
            q.put(('status', "Reading .hspy…"))
            if cancel_flag['stop']: return
            obj = rs_hspy.file_reader(path)  # may be dict, list, or ndarray
            if cancel_flag['stop']: return
            try:
                arr, dbg = _extract_rsciio_array(obj)
            except Exception as e:
                q.put(('error', f"Extract failed: {e}"));
                return
            q.put(('status', f"Parsed: {dbg}"))
            q.put(('done', arr))

        _load_with_progress(start_fn, "Load .hspy")

    def load_data():
        """
        Loads a .npy file as a 4D data array.
        Accepts both 4D (Y, X, H, W) and 3D (N, H, W) stacks.
        If 3D, asks user for scan_width to reshape.
        """
        nonlocal data_array, circle_center
        file_path = filedialog.askopenfilename(parent=root, filetypes=[("NumPy Array", "*.npy")])
        if not file_path:
            return

        arr = load_npy_with_progress(file_path)
        if arr.ndim == 4:
            # Already correct
            data_array = arr

        elif arr.ndim == 3:
            N, H, W = arr.shape
            guessed_scan_width = int(np.sqrt(N))
            scan_width = ask_scan_width(root, guessed_scan_width, N)
            if not scan_width:
                return
            if N % scan_width != 0:
                messagebox.showwarning(
                    "Load NumPy Array",
                    f"{N} frames not divisible by {scan_width}; last row will be padded.",
                    parent=root
                )
                missing = scan_width - (N % scan_width)
                pad = np.zeros((missing, H, W), dtype=arr.dtype)
                arr = np.concatenate([arr, pad], axis=0)
                N = arr.shape[0]

            scan_height = N // scan_width
            data_array = arr.reshape(scan_height, scan_width, H, W)

        else:
            messagebox.showerror("Load NumPy Array",
                                 f"Expected 3D or 4D array, got shape {arr.shape}",
                                 parent=root)
            return

        # Center integration mask + preview
        H, W = data_array.shape[2], data_array.shape[3]
        circle_center[0], circle_center[1] = W // 2, H // 2
        update_pointer_image(data_array.shape[1] // 2, data_array.shape[0] // 2)

        messagebox.showinfo("Import Complete",
                            f"Loaded array with shape {data_array.shape[:2]} navigator.",
                            parent=root)
        update_function()

    def update_function(*args):
        """
        Updates the main image based on the selected function.
        Passes both the radius and the circle center to the user function.
        """
        nonlocal main_image_pil
        if data_array is None or selected_function.get() not in functions:
            return
        function = functions[selected_function.get()]

        # Ensure the circle center is valid
        #if circle_center[0] is None or circle_center[1] is None:
        #    circle_center[0] = 0
        #    circle_center[1] = 0
        if circle_center[0] is None or circle_center[1] is None:
            H, W = data_array.shape[2], data_array.shape[3]  # diffraction frame (rows, cols)
            circle_center[0] = W // 2  # x
            circle_center[1] = H // 2  # y

        # Call the user function with radius and center
        main_image = function(data_array, radius_value.get(), tuple(circle_center))
        main_image_pil = Image.fromarray(normalize_to_8bit(main_image))
        update_pointer_image(data_array.shape[1] // 2, data_array.shape[0] // 2)  # args: (x, y)
        update_main_image()

    def export_blo():
        """Prompt for pixel size + path, then write current data_array to .blo."""
        if data_array is None:
            messagebox.showwarning("Export .blo", "No data loaded.", parent=root);
            return

        px = ask_pixel_size_nm(root, guessed_nm=None)
        if px is None:  # cancelled
            return

        path = filedialog.asksaveasfilename(
            parent=root,
            defaultextension=".blo",
            filetypes=[("Blockfile", "*.blo"), ("All files", "*.*")],
            initialfile="exported.blo",
            title="Save .blo"
        )
        if not path:
            return

        try:
            # You can set bin_data_to_128=False if you want native DP size preserved.
            array2blo(data_array, px_nm=px, save_path=path,
                      intensity_scaling="crop", bin_data_to_128=True)
        except Exception as e:
            messagebox.showerror("Export .blo", f"Failed to write file:\n{e}", parent=root);
            return

        messagebox.showinfo("Export .blo", f"Saved:\n{path}", parent=root)

    def export_tiff_folder():
        """Threaded, cancellable export of each DP as 00001.tiff, 00002.tiff, ..."""
        if data_array is None:
            messagebox.showwarning("Export TIFFs", "No data loaded.", parent=root)
            return

        out_dir = filedialog.askdirectory(parent=root, title="Select output folder")
        if not out_dir:
            return

        Y, X, H, W = data_array.shape
        total = Y * X
        pad = len(str(total))  # zero-padding width

        # Progress window (modal)
        win = tk.Toplevel(root)
        win.title("Export TIFFs")
        win.transient(root)
        win.grab_set()
        tk.Label(win, text=f"Exporting {total} TIFFs…").pack(padx=10, pady=(10, 0))
        bar = ttk.Progressbar(win, orient="horizontal", length=360, mode="determinate", maximum=total)
        bar.pack(padx=10, pady=10)
        lbl = tk.Label(win, text=f"0 / {total} (0.0%)")
        lbl.pack(padx=10, pady=(0, 10))
        cancel_flag = {'stop': False}

        def _cancel():
            cancel_flag['stop'] = True
            lbl.config(text="Cancelling…")

        tk.Button(win, text="Cancel", command=_cancel).pack(pady=(0, 10))

        q = queue.Queue()

        def worker():
            try:
                idx = 0
                # Ensures directory exists
                os.makedirs(out_dir, exist_ok=True)
                for y in range(Y):
                    if cancel_flag['stop']:
                        q.put(('cancelled', None));
                        return
                    for x in range(X):
                        if cancel_flag['stop']:
                            q.put(('cancelled', None));
                            return
                        idx += 1
                        fname = os.path.join(out_dir, f"{idx:0{pad}d}.tiff")
                        # Write one DP (original dtype preserved)
                        tiff.imwrite(fname, data_array[y, x], photometric='minisblack')
                        # Post progress
                        if idx % 1 == 0:
                            q.put(('progress', idx))
                q.put(('done', total))
            except Exception as e:
                q.put(('error', f"{type(e).__name__}: {e}"))

        def pump():
            try:
                while True:
                    kind, payload = q.get_nowait()
                    if kind == 'progress':
                        i = payload
                        bar['value'] = i
                        lbl.config(text=f"{i} / {total} ({(i / total) * 100:.1f}%)")
                    elif kind == 'done':
                        win.grab_release();
                        win.destroy()
                        messagebox.showinfo("Export TIFFs", f"Exported {total} TIFFs to:\n{out_dir}", parent=root)
                        return
                    elif kind == 'cancelled':
                        win.grab_release();
                        win.destroy()
                        messagebox.showinfo("Export TIFFs", "Export cancelled.", parent=root)
                        return
                    elif kind == 'error':
                        win.grab_release();
                        win.destroy()
                        messagebox.showerror("Export TIFFs", payload, parent=root)
                        return
            except queue.Empty:
                pass
            root.after(33, pump)  # ~30 Hz UI updates

        threading.Thread(target=worker, daemon=True).start()
        pump()

    # UI Layout


    # Set the desired square canvas size
    SQUARE_CANVAS_SIZE = 512  # Example size

    # Top Frame for Controls
    top_frame = tk.Frame(root)
    top_frame.pack(side=tk.TOP, fill=tk.X)

    # Load Data Button
    load_button = tk.Button(top_frame, text="Load Numpy Array", command=load_data)
    load_button.pack(side=tk.LEFT, padx=5, pady=5)

    #Save images button
    save_images = tk.Button(top_frame, text="Save Images", command=save_images)
    save_images.pack(side=tk.LEFT, padx=5, pady=5)

    #Load Series button
    load_series_b = tk.Button(top_frame, text="Load .tiff series", command=load_series)
    load_series_b.pack(side=tk.LEFT, padx=5, pady=5)


    #blo reader
    tk.Button(top_frame, text="Load .blo", command=load_blo).pack(side=tk.LEFT, padx=5, pady=5)
    #hspy reader
    tk.Button(top_frame, text="Load .hspy", command=load_hspy).pack(side=tk.LEFT, padx=5, pady=5)

    export_blo_btn = tk.Button(top_frame, text="Export .blo", command=export_blo)
    export_blo_btn.pack(side=tk.LEFT, padx=5, pady=5)

    export_tif_btn = tk.Button(top_frame, text="Export TIFFs", command=export_tiff_folder)
    export_tif_btn.pack(side=tk.LEFT, padx=5, pady=5)

    # Radius Label and Entry
    radius_label = tk.Label(top_frame, text="Detector Radius (px):")
    radius_label.pack(side=tk.LEFT, padx=5, pady=5)

    radius_entry = tk.Entry(top_frame, textvariable=radius_value, width=5)
    radius_entry.pack(side=tk.LEFT, padx=5, pady=5)
    radius_entry.bind("<Return>", update_function)

    # Function Dropdown Menu
    function_dropdown = ttk.Combobox(top_frame, textvariable=selected_function, values=list(functions.keys()),
                                     state="readonly")
    function_dropdown.pack(side=tk.LEFT, padx=5, pady=5)
    function_dropdown.bind("<<ComboboxSelected>>", update_function)

    # Main Frame for Canvases
    main_frame = tk.Frame(root)
    main_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    # Left Canvas (Main Image)
    main_canvas = tk.Canvas(main_frame, width=1024, height=1024, bg="white")
    main_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    main_canvas.bind("<Button-1>", on_click)
    main_canvas.bind("<Motion>", _on_mouse_motion)

    # Right Canvas (Pointer Image)
    pointer_canvas = tk.Canvas(main_frame, width=SQUARE_CANVAS_SIZE, height=SQUARE_CANVAS_SIZE, bg="white")
    pointer_canvas.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
    pointer_canvas.bind("<Button-1>", on_pointer_click)
    main_canvas.bind("<Button-2>", toggle_mouse_motion)  # Right-click to toggle mouse motion functionality

    # Set the initial size for canvases
    main_canvas.config(width=1024, height=1024)
    pointer_canvas.config(width=SQUARE_CANVAS_SIZE, height=SQUARE_CANVAS_SIZE)

    # Start the Tkinter main loop
    root.mainloop()

#visualiser()
if __name__ == "__main__":
    visualiser()