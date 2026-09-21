"""Interactive in-gel fluorescence densitometry."""

from __future__ import annotations

import csv
import json
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import matplotlib
import numpy as np
import tifffile
from PIL import Image
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.widgets import RectangleSelector, SpanSelector

matplotlib.use("TkAgg")


class DensitometryApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("In-gel fluorescence densitometry")
        self.root.geometry("1250x900")

        self.image: np.ndarray | None = None
        self.original_image: np.ndarray | None = None
        self.image_artist = None
        self.roi_bounds: tuple[int, int, int, int] | None = None
        self.baseline_points: list[tuple[float, float]] = []
        self.peaks: list[tuple[float, float]] = []
        self.roi_selector: RectangleSelector | None = None
        self.crop_selector: RectangleSelector | None = None
        self.span_selector: SpanSelector | None = None
        self.background_connection_id: int | None = None
        self.auc_click_connection_id: int | None = None
        self.copy_connection_id: int | None = None
        self.vertical_line_connection_id: int | None = None
        self.plot_marker_lines = []
        self.image_marker_lines = []
        self.plot_marker_positions: list[float] = []
        self.undo_actions = []
        self.profile_data_by_axis = {}
        self.roi_bounds_by_axis = {}
        self.background_lines = []
        self.baseline_points_by_axis = {}
        self.roi_motion_connection_id: int | None = None
        self.roi_press_connection_id: int | None = None
        self.roi_release_connection_id: int | None = None
        self.roi_dragging = False
        self.roi_drag_start: tuple[float, float] | None = None
        self.roi_drag_origin: tuple[int, int] | None = None
        self.roi_last_position: tuple[int, int] | None = None
        self.profile_update_after_id = None
        self.roi_alignment_x: int | None = None
        self.roi_patch = None
        self.roi_label = None
        self.roi_patches = []
        self.roi_labels = []
        self.roi_count = 0
        self.current_roi_number = 0
        self.auc_artists = []
        self.auc_results = []
        self.profile_auc_texts = {}
        self.mode = tk.StringVar(value="Open a TIFF to begin")
        self.brightness = tk.DoubleVar(value=0.0)
        self.contrast = tk.DoubleVar(value=1.0)
        self.rotation_angle = tk.StringVar(value="0")

        self._build_ui()

    def _build_ui(self) -> None:
        toolbar = ttk.Frame(self.root, padding=8)
        toolbar.pack(fill=tk.X)
        ttk.Button(toolbar, text="Open TIFF", command=self.open_tiff).pack(side=tk.LEFT)
        ttk.Button(toolbar, text="Open Session", command=self.open_session).pack(side=tk.LEFT, padx=5)
        ttk.Button(toolbar, text="Save Session", command=self.save_session).pack(side=tk.LEFT)
        ttk.Button(toolbar, text="Crop image view", command=self.enable_crop).pack(side=tk.LEFT, padx=5)
        ttk.Button(toolbar, text="Reset image view", command=self.reset_image_view).pack(side=tk.LEFT)
        ttk.Button(toolbar, text="Draw ROI", command=self.enable_roi).pack(side=tk.LEFT, padx=5)
        self.copy_button = ttk.Button(toolbar, text="Copy ROI", command=self.enable_copy_roi)
        self.copy_button.pack(side=tk.LEFT)
        ttk.Button(
            toolbar, text="Mark Profile Boundaries", command=self.enable_vertical_line
        ).pack(side=tk.LEFT, padx=5)
        ttk.Button(
            toolbar, text="Draw Background", command=self.connect_marker_background
        ).pack(side=tk.LEFT)
        ttk.Button(toolbar, text="Magic Wand", command=self.enable_wand).pack(side=tk.LEFT, padx=5)
        ttk.Button(toolbar, text="Undo", command=self.undo).pack(side=tk.LEFT)
        ttk.Button(toolbar, text="Reset", command=self.reset).pack(side=tk.LEFT)
        ttk.Button(toolbar, text="Export AUCs", command=self.export_auc_csv).pack(side=tk.RIGHT, padx=(0, 5))
        ttk.Button(toolbar, text="Export Profile", command=self.export_csv).pack(side=tk.RIGHT)

        display_controls = ttk.Frame(self.root, padding=(8, 0, 8, 4))
        display_controls.pack(fill=tk.X)
        ttk.Label(display_controls, text="Display brightness").pack(side=tk.LEFT)
        ttk.Scale(
            display_controls, from_=-1.0, to=1.0, variable=self.brightness,
            command=self._update_display,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(6, 18))
        ttk.Label(display_controls, text="Display contrast").pack(side=tk.LEFT)
        ttk.Scale(
            display_controls, from_=0.1, to=3.0, variable=self.contrast,
            command=self._update_display,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)
        ttk.Label(display_controls, text="Rotation (degrees)").pack(side=tk.LEFT, padx=(12, 4))
        rotation_entry = ttk.Entry(
            display_controls, textvariable=self.rotation_angle, width=8,
        )
        rotation_entry.pack(side=tk.LEFT)
        rotation_entry.bind("<Return>", lambda _event: self.apply_rotation())
        ttk.Button(
            display_controls, text="Apply rotation", command=self.apply_rotation
        ).pack(side=tk.LEFT, padx=(4, 0))
        ttk.Button(
            display_controls, text="Reset rotation", command=self.reset_rotation
        ).pack(side=tk.LEFT, padx=(4, 0))
        ttk.Button(
            display_controls, text="Auto display range", command=self.reset_display_range
        ).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Label(
            display_controls,
            text="Display only: measurements always use the original grey values.",
        ).pack(side=tk.LEFT, padx=(12, 0))
        for key in ("<Up>", "<Down>", "<Left>", "<Right>"):
            self.root.bind_all(key, self._arrow_key_move_roi)
        self.root.bind_all("<Delete>", self._delete_key_remove_roi)

        self.figure = Figure(figsize=(11, 8), dpi=100)
        self.image_axis = self.figure.add_subplot(121)
        self.profile_axis = self.figure.add_subplot(122)
        self.profile_axes = [self.profile_axis]
        self.marker_summary_axis = self.figure.add_axes([0.80, 0.08, 0.19, 0.86])
        self.marker_summary_axis.set_title("Profile and AUC", fontsize=9)
        self.marker_summary_axis.axis("off")
        self.image_axis.set_title("Open a grayscale TIFF")
        self.profile_axis.set_title("Plot profile P(x)")
        self.profile_axis.set_xlabel("Horizontal distance (pixels)")
        self.profile_axis.set_ylabel("Mean grey value")
        self.figure.subplots_adjust(left=0.06, right=0.98, top=0.94, bottom=0.08, wspace=0.28)
        plot_frame = ttk.Frame(self.root)
        plot_frame.pack(fill=tk.BOTH, expand=True)
        self.plot_scrollbar = ttk.Scrollbar(plot_frame, orient=tk.VERTICAL)
        self.plot_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.plot_scroll = tk.Canvas(
            plot_frame, highlightthickness=0, yscrollcommand=self.plot_scrollbar.set
        )
        self.plot_scroll.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.plot_scrollbar.configure(command=self.plot_scroll.yview)
        self.plot_host = ttk.Frame(self.plot_scroll)
        self.plot_window = self.plot_scroll.create_window(
            (0, 0), window=self.plot_host, anchor="nw"
        )
        self.plot_host.bind("<Configure>", self._update_plot_scrollregion)
        self.plot_scroll.bind("<Configure>", self._resize_plot_host)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.plot_host)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas.get_tk_widget().bind("<MouseWheel>", self._scroll_plot)
        self.plot_scroll.bind("<MouseWheel>", self._scroll_plot)
        self.roi_motion_connection_id = self.canvas.mpl_connect(
            "motion_notify_event", self._roi_motion
        )
        self.roi_press_connection_id = self.canvas.mpl_connect(
            "button_press_event", self._roi_press
        )
        self.roi_release_connection_id = self.canvas.mpl_connect(
            "button_release_event", self._roi_release
        )
        self._layout_profile_axes()
        self.copy_button.configure(state=tk.DISABLED)

    def _update_plot_scrollregion(self, _event=None) -> None:
        self.plot_scroll.configure(scrollregion=self.plot_scroll.bbox("all"))

    def _resize_plot_host(self, event) -> None:
        self.plot_scroll.itemconfigure(self.plot_window, width=event.width)

    def _scroll_plot(self, event) -> str:
        if event.delta:
            self.plot_scroll.yview_scroll(-int(event.delta / 120), "units")
        return "break"

    def open_tiff(self) -> None:
        path = filedialog.askopenfilename(
            title="Open grayscale TIFF",
            filetypes=[("TIFF files", "*.tif *.tiff"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            image = np.asarray(tifffile.imread(path))
            if image.ndim > 2:
                image = image[0]
            if image.ndim != 2:
                raise ValueError("The TIFF must contain a 2D grayscale image.")
            self.image = image.astype(float)
            self.original_image = self.image.copy()
        except Exception as exc:
            messagebox.showerror("Could not open TIFF", str(exc))
            return

        self.reset(clear_image=False)
        self.image_axis.clear()
        self.image_artist = self.image_axis.imshow(
            self.image, cmap="gray", aspect="equal", origin="upper"
        )
        self.image_axis.set_title(Path(path).name)
        self.image_axis.set_xlabel("x (pixels)")
        self.image_axis.set_ylabel("y (pixels)")
        self._update_display()
        self.canvas.draw_idle()
        self.mode.set("Select a horizontal lane ROI on the image")

    def save_session(self) -> None:
        if self.image is None:
            messagebox.showinfo("Nothing to save", "Open a TIFF before saving a session.")
            return
        path = filedialog.asksaveasfilename(
            title="Save densitometry session",
            defaultextension=".npz",
            filetypes=[("Densitometry sessions", "*.npz"), ("All files", "*.*")],
        )
        if not path:
            return

        axis_indexes = {axis: index for index, axis in enumerate(self.profile_axes)}
        metadata = {
            "version": 1,
            "image_title": self.image_axis.get_title(),
            "rotation_angle": self.rotation_angle.get(),
            "brightness": self.brightness.get(),
            "contrast": self.contrast.get(),
            "image_xlim": [float(value) for value in self.image_axis.get_xlim()],
            "image_ylim": [float(value) for value in self.image_axis.get_ylim()],
            "roi_bounds": [
                list(self.roi_bounds_by_axis[axis])
                for axis in self.profile_axes
                if axis in self.roi_bounds_by_axis
            ],
            "roi_alignment_x": self.roi_alignment_x,
            "plot_marker_positions": [float(value) for value in self.plot_marker_positions],
            "baseline_points": [
                [[float(x), float(y)] for x, y in self.baseline_points_by_axis.get(axis, [])]
                for axis in self.profile_axes
            ],
            "peaks": [[float(start), float(end)] for start, end in self.peaks],
            "auc_results": [
                {
                    "axis": axis_indexes[result["axis"]],
                    "number": result["number"],
                    "start": result["start"],
                    "end": result["end"],
                    "profile_auc": result["profile_auc"],
                    "background_auc": result["background_auc"],
                    "corrected_auc": result["corrected_auc"],
                }
                for result in self.auc_results
            ],
        }
        try:
            np.savez_compressed(
                path,
                image=self.image,
                original_image=self.original_image if self.original_image is not None else self.image,
                metadata=json.dumps(metadata),
            )
        except Exception as exc:
            messagebox.showerror("Could not save session", str(exc))
            return
        self.mode.set(f"Saved session: {Path(path).name}")

    def open_session(self) -> None:
        path = filedialog.askopenfilename(
            title="Open densitometry session",
            filetypes=[("Densitometry sessions", "*.npz"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            with np.load(path, allow_pickle=False) as session:
                image = np.asarray(session["image"], dtype=float)
                original_image = np.asarray(session["original_image"], dtype=float)
                metadata = json.loads(str(session["metadata"].item()))
            if image.ndim != 2 or original_image.ndim != 2:
                raise ValueError("The session does not contain a 2D grayscale image.")
            roi_bounds = metadata.get("roi_bounds", [])
            baseline_points = metadata.get("baseline_points", [])
            marker_positions = metadata.get("plot_marker_positions", [])
            auc_results = metadata.get("auc_results", [])
        except Exception as exc:
            messagebox.showerror("Could not open session", str(exc))
            return

        self.image = image
        self.original_image = original_image
        self.reset(clear_image=False)
        self.image_axis.clear()
        self.image_artist = self.image_axis.imshow(
            self.image, cmap="gray", aspect="equal", origin="upper"
        )
        self.image_axis.set_title(metadata.get("image_title") or Path(path).name)
        self.image_axis.set_xlabel("x (pixels)")
        self.image_axis.set_ylabel("y (pixels)")
        self.rotation_angle.set(metadata.get("rotation_angle", "0"))
        self.brightness.set(float(metadata.get("brightness", 0.0)))
        self.contrast.set(float(metadata.get("contrast", 1.0)))

        for bounds in roi_bounds:
            if len(bounds) != 4:
                raise ValueError("The session contains invalid ROI bounds.")
            self._set_roi(
                *[int(value) for value in bounds],
                new_profile=True,
                preserve_alignment=False,
            )
        self.roi_alignment_x = metadata.get("roi_alignment_x")

        for index, points in enumerate(baseline_points[:len(self.profile_axes)]):
            axis = self.profile_axes[index]
            parsed_points = [(float(x), float(y)) for x, y in points]
            self.baseline_points_by_axis[axis] = parsed_points
            if parsed_points:
                background_line, = axis.plot(
                    [point[0] for point in parsed_points],
                    [point[1] for point in parsed_points],
                    color="#dc2626", linestyle="--", zorder=1,
                )
                self.background_lines.append(background_line)
            if axis is self.profile_axis:
                self.baseline_points = parsed_points.copy()

        for marker_number, x_position in enumerate(marker_positions, 1):
            marker_group = []
            image_marker_group = []
            for axis in self.profile_axes:
                axis_min, axis_max = sorted(axis.get_xlim())
                axis_position = min(max(float(x_position), axis_min), axis_max)
                line = axis.axvline(axis_position, color="#7c3aed", linewidth=1.5)
                marker_text = axis.text(
                    axis_position, 0.98, f"x{marker_number}",
                    transform=axis.get_xaxis_transform(), color="#7c3aed",
                    ha="center", va="top", fontsize=9,
                )
                marker_group.append((axis, line, marker_text))
                roi = self.roi_bounds_by_axis.get(axis)
                if roi is not None:
                    image_line = self.image_axis.axvline(
                        roi[0] + axis_position, color="#7c3aed", linewidth=1.2, linestyle="--", alpha=0.85
                    )
                    image_text = self.image_axis.text(
                        roi[0] + axis_position, 0.98, f"x{marker_number}",
                        transform=self.image_axis.get_xaxis_transform(), color="#7c3aed",
                        ha="center", va="top", fontsize=9,
                        bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none", "pad": 1},
                    )
                    image_marker_group.append((image_line, image_text))
            self.plot_marker_positions.append(float(x_position))
            self.plot_marker_lines.extend(
                (line, marker_text) for _, line, marker_text in marker_group
            )
            self.image_marker_lines.extend(image_marker_group)

        for saved_result in auc_results:
            axis_index = int(saved_result["axis"])
            if axis_index >= len(self.profile_axes):
                continue
            axis = self.profile_axes[axis_index]
            computed = self._compute_auc(
                axis,
                (float(saved_result["start"]), float(saved_result["end"])),
                show_popup=False,
            )
            if computed is not None:
                self.auc_results[-1]["number"] = int(saved_result["number"])

        self.peaks = [tuple(float(value) for value in peak) for peak in metadata.get("peaks", [])]
        for peak in self.peaks:
            self.profile_axis.axvspan(*peak, color="#16a34a", alpha=0.2)
        image_xlim = metadata.get("image_xlim")
        image_ylim = metadata.get("image_ylim")
        if image_xlim and image_ylim:
            self.image_axis.set_xlim(*image_xlim)
            self.image_axis.set_ylim(*image_ylim)
        self._update_display()
        self._update_marker_summary()
        self.canvas.draw_idle()
        self.mode.set(f"Opened session: {Path(path).name}")

    def apply_rotation(self) -> None:
        if self.original_image is None:
            messagebox.showinfo("Open TIFF", "Open a TIFF image first.")
            return
        try:
            angle = float(self.rotation_angle.get().strip())
        except (tk.TclError, ValueError):
            messagebox.showerror("Invalid rotation", "Enter an angle between -180 and 180 degrees.")
            return
        if not -180.0 <= angle <= 180.0:
            messagebox.showerror("Invalid rotation", "Enter an angle between -180 and 180 degrees.")
            return

        rotated = Image.fromarray(self.original_image.astype(np.float32), mode="F").rotate(
            angle, resample=Image.Resampling.BICUBIC, expand=True
        )
        self.image = np.asarray(rotated, dtype=float)
        self.reset(clear_image=False)
        self.image_axis.clear()
        self.image_artist = self.image_axis.imshow(
            self.image, cmap="gray", aspect="equal", origin="upper"
        )
        self.image_axis.set_title("Rotated gel")
        self.image_axis.set_xlabel("x (pixels)")
        self.image_axis.set_ylabel("y (pixels)")
        self._update_display()
        self.canvas.draw_idle()
        self.mode.set(f"Gel rotated {angle:g} degrees. Select a horizontal lane ROI")

    def reset_rotation(self) -> None:
        if self.original_image is None:
            return
        self.rotation_angle.set(0.0)
        self.apply_rotation()

    def enable_crop(self) -> None:
        if self.image is None:
            messagebox.showinfo("Open TIFF", "Open a TIFF image first.")
            return
        self._disconnect_selectors()
        self.mode.set("Drag a rectangle to crop the image view")
        self.crop_selector = RectangleSelector(
            self.image_axis,
            self._crop_selected,
            useblit=True,
            button=[1],
            minspanx=2,
            minspany=2,
            interactive=True,
        )

    def _crop_selected(self, click, release) -> None:
        if self.image is None or click.xdata is None or release.xdata is None:
            return
        x0, x1 = sorted((click.xdata, release.xdata))
        y0, y1 = sorted((click.ydata, release.ydata))
        x0 = max(0, min(self.image.shape[1], x0))
        x1 = max(0, min(self.image.shape[1], x1))
        y0 = max(0, min(self.image.shape[0], y0))
        y1 = max(0, min(self.image.shape[0], y1))
        if x1 - x0 < 2 or y1 - y0 < 2:
            return
        self.image_axis.set_xlim(x0, x1)
        self.image_axis.set_ylim(y1, y0)
        self._disconnect_selectors()
        self.canvas.draw_idle()
        self.mode.set("Image view cropped. Select or copy an ROI")

    def reset_image_view(self) -> None:
        if self.image is None:
            return
        self._disconnect_selectors()
        self.image_axis.set_xlim(-0.5, self.image.shape[1] - 0.5)
        self.image_axis.set_ylim(self.image.shape[0] - 0.5, -0.5)
        self.canvas.draw_idle()
        self.mode.set("Full image view restored")

    def enable_vertical_line(self) -> None:
        if self.profile is None:
            messagebox.showinfo("Select lane", "Select a lane ROI or draw a lane first.")
            return
        self._disconnect_selectors()
        self.mode.set("Click the lane plot to add x1, then x2")
        self.vertical_line_connection_id = self.canvas.mpl_connect(
            "button_press_event", self._vertical_line_click
        )

    def _vertical_line_click(self, event) -> None:
        if self.profile is None or event.inaxes not in self.profile_axes or event.xdata is None:
            return
        target_axis = event.inaxes
        x_min, x_max = target_axis.get_xlim()
        x_position = min(max(event.xdata, min(x_min, x_max)), max(x_min, x_max))
        marker_number = len(self.plot_marker_positions) + 1
        marker_group = []
        image_marker_group = []
        for axis in self.profile_axes:
            axis_min, axis_max = axis.get_xlim()
            axis_position = min(max(x_position, min(axis_min, axis_max)), max(axis_min, axis_max))
            line = axis.axvline(axis_position, color="#7c3aed", linewidth=1.5)
            marker_text = axis.text(
                axis_position, 0.98, f"x{marker_number}",
                transform=axis.get_xaxis_transform(),
                color="#7c3aed", ha="center", va="top", fontsize=9,
            )
            marker_group.append((axis, line, marker_text))
            roi_bounds = self.roi_bounds_by_axis.get(axis)
            if roi_bounds is not None:
                image_x = roi_bounds[0] + axis_position
                image_line = self.image_axis.axvline(
                    image_x, color="#7c3aed", linewidth=1.2, linestyle="--", alpha=0.85
                )
                image_text = self.image_axis.text(
                    image_x, 0.98, f"x{marker_number}",
                    transform=self.image_axis.get_xaxis_transform(),
                    color="#7c3aed", ha="center", va="top", fontsize=9,
                    bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none", "pad": 1},
                )
                image_marker_group.append((image_line, image_text))
        self.plot_marker_positions.append(x_position)
        self.plot_marker_lines.extend(
            (line, marker_text) for _, line, marker_text in marker_group
        )
        self.image_marker_lines.extend(image_marker_group)
        self.undo_actions.append(("marker_group", marker_group, image_marker_group))
        self._update_marker_summary()
        self.canvas.draw_idle()
        self.mode.set(
            "x1 marked. Click again for x2" if marker_number == 1
            else "x2 marked. Draw another line or use Undo"
        )

    def enable_roi(self) -> None:
        if self.image is None:
            messagebox.showinfo("Open TIFF", "Open a TIFF image first.")
            return
        self._disconnect_selectors()
        self.mode.set("Drag across the lane ROI")
        self.roi_selector = RectangleSelector(
            self.image_axis,
            self._roi_selected,
            useblit=True,
            button=[1],
            minspanx=2,
            minspany=2,
            interactive=True,
        )

    def enable_copy_roi(self) -> None:
        if self.image is None or self.roi_bounds is None:
            messagebox.showinfo("Select ROI", "Select a lane ROI first.")
            return
        original_x0, original_y0, width, height = self.roi_bounds
        x0 = original_x0
        y0 = max(0, original_y0 - 170)
        self._set_roi(x0, y0, width, height, new_profile=True)
        self.mode.set("ROI copied 170 pixels above. Copy again or select a peak")

    def _roi_selected(self, click, release) -> None:
        if self.image is None:
            return
        x0, x1 = sorted((int(round(click.xdata)), int(round(release.xdata))))
        y0, y1 = sorted((int(round(click.ydata)), int(round(release.ydata))))
        x0, x1 = max(0, x0), min(self.image.shape[1], x1 + 1)
        y0, y1 = max(0, y0), min(self.image.shape[0], y1 + 1)
        if x1 <= x0 or y1 <= y0:
            return
        self._set_roi(x0, y0, x1 - x0, y1 - y0, new_profile=True)
        self._disconnect_selectors()

    def _roi_edge_hit(self, event) -> bool:
        if self.image is None or self.roi_bounds is None:
            return False
        if event.inaxes != self.image_axis or event.xdata is None or event.ydata is None:
            return False
        x0, y0, width, height = self.roi_bounds
        tolerance = max(3.0, min(width, height) * 0.04)
        near_x_edge = abs(event.xdata - x0) <= tolerance or abs(event.xdata - (x0 + width)) <= tolerance
        near_y_edge = abs(event.ydata - y0) <= tolerance or abs(event.ydata - (y0 + height)) <= tolerance
        inside_x = x0 - tolerance <= event.xdata <= x0 + width + tolerance
        inside_y = y0 - tolerance <= event.ydata <= y0 + height + tolerance
        return (near_x_edge and inside_y) or (near_y_edge and inside_x)

    def _roi_motion(self, event) -> None:
        if self.roi_dragging:
            if event.inaxes != self.image_axis or event.xdata is None or event.ydata is None:
                return
            if self.roi_drag_start is None or self.roi_drag_origin is None or self.roi_bounds is None:
                return
            dx = int(round(event.xdata - self.roi_drag_start[0]))
            dy = int(round(event.ydata - self.roi_drag_start[1]))
            position = (self.roi_drag_origin[0] + dx, self.roi_drag_origin[1] + dy)
            if position == self.roi_last_position:
                return
            self.roi_last_position = position
            _, _, width, height = self.roi_bounds
            self._set_roi(
                position[0],
                position[1],
                width,
                height,
                new_profile=False,
                update_profile=False,
            )
            self._schedule_profile_update()
            self.canvas.get_tk_widget().configure(cursor="fleur")
            return
        cursor = "hand2" if self._roi_edge_hit(event) else ""
        self.canvas.get_tk_widget().configure(cursor=cursor)

    def _roi_press(self, event) -> None:
        if event.button != 1 or self.roi_bounds is None:
            return
        x0, y0, _, _ = self.roi_bounds
        if event.inaxes != self.image_axis or event.xdata is None or event.ydata is None:
            return
        inside_roi = x0 <= event.xdata <= x0 + self.roi_bounds[2] and y0 <= event.ydata <= y0 + self.roi_bounds[3]
        if not inside_roi:
            return
        self.canvas.get_tk_widget().focus_set()
        if not self._roi_edge_hit(event):
            return
        self.roi_dragging = True
        self.roi_drag_start = (event.xdata, event.ydata)
        self.roi_drag_origin = (x0, y0)
        self.roi_last_position = (x0, y0)
        self.canvas.get_tk_widget().configure(cursor="fleur")

    def _roi_release(self, event) -> None:
        if event.button != 1:
            return
        if self.roi_dragging and self.roi_bounds is not None and self.current_roi_number > 1:
            x0, y0, width, height = self.roi_bounds
            if self.roi_alignment_x is not None and x0 != self.roi_alignment_x:
                self._set_roi(
                    self.roi_alignment_x,
                    y0,
                    width,
                    height,
                    new_profile=False,
                    update_profile=False,
                )
                self._schedule_profile_update()
        self.roi_dragging = False
        self.roi_drag_start = None
        self.roi_drag_origin = None
        self.roi_last_position = None
        self.canvas.get_tk_widget().configure(cursor="")

    def _nudge_roi(self, dx: int, dy: int) -> None:
        if self.image is None or self.roi_bounds is None:
            return
        x0, y0, width, height = self.roi_bounds
        self._set_roi(
            x0 + dx,
            y0 + dy,
            width,
            height,
            new_profile=False,
            update_profile=True,
            preserve_alignment=False,
        )
        direction = {(-1, 0): "left", (1, 0): "right", (0, -1): "up", (0, 1): "down"}
        self.mode.set(f"ROI moved {direction[(dx, dy)]}")

    def _arrow_key_move_roi(self, event):
        if event.widget.winfo_class() in {"Entry", "TEntry", "TSpinbox"}:
            return
        movement = {
            "Up": (0, -1),
            "Down": (0, 1),
            "Left": (-1, 0),
            "Right": (1, 0),
        }.get(event.keysym)
        if movement is None or self.roi_bounds is None:
            return
        self._nudge_roi(*movement)
        return "break"

    def _delete_key_remove_roi(self, event):
        if event.widget.winfo_class() in {"Entry", "TEntry", "TSpinbox"}:
            return
        if self.roi_bounds is None:
            return
        self._delete_active_roi()
        return "break"

    def _delete_active_roi(self) -> None:
        deleted_axis = self.profile_axis
        if self.roi_patch is not None:
            self.roi_patch.remove()
            if self.roi_patch in self.roi_patches:
                self.roi_patches.remove(self.roi_patch)
        if self.roi_label is not None:
            self.roi_label.remove()
            if self.roi_label in self.roi_labels:
                self.roi_labels.remove(self.roi_label)

        for line, marker_text in self.plot_marker_lines[:]:
            if line.axes is deleted_axis:
                line.remove()
                marker_text.remove()
                self.plot_marker_lines.remove((line, marker_text))
        for line in self.background_lines[:]:
            if line.axes is deleted_axis:
                line.remove()
                self.background_lines.remove(line)
        for artist in self.auc_artists[:]:
            if getattr(artist, "axes", None) is deleted_axis:
                artist.remove()
                self.auc_artists.remove(artist)
        profile_auc_text = self.profile_auc_texts.pop(deleted_axis, None)
        if profile_auc_text is not None and profile_auc_text.axes is not None:
            profile_auc_text.remove()
        self.auc_results = [result for result in self.auc_results if result["axis"] is not deleted_axis]
        self.profile_data_by_axis.pop(deleted_axis, None)
        self.roi_bounds_by_axis.pop(deleted_axis, None)
        self.baseline_points_by_axis.pop(deleted_axis, None)

        if len(self.profile_axes) > 1:
            self.profile_axes.remove(deleted_axis)
            deleted_axis.remove()
            self.profile_axis = self.profile_axes[-1]
            self.roi_bounds = self.roi_bounds_by_axis.get(self.profile_axis)
            profile_data = self.profile_data_by_axis.get(self.profile_axis)
            self.profile = profile_data[1].copy() if profile_data is not None else None
            self.roi_patch = self.roi_patches[-1] if self.roi_patches else None
            self.roi_label = self.roi_labels[-1] if self.roi_labels else None
            self.current_roi_number = len(self.profile_axes)
            self._layout_profile_axes()
        else:
            self.profile = None
            self.roi_bounds = None
            self.roi_patch = None
            self.roi_label = None
            self.current_roi_number = 0
            self.profile_axis.clear()
            self.profile_axis.set_title("Plot profile P(x)")
            self.profile_axis.set_xlabel("Horizontal distance (pixels)")
            self.profile_axis.set_ylabel("Mean grey value")
        self.roi_count = len(self.roi_patches)
        self._update_marker_summary()
        self.canvas.draw_idle()
        self.mode.set("ROI and its plot profile removed")

    def _set_roi(
        self,
        x0: int,
        y0: int,
        width: int,
        height: int,
        new_profile: bool = True,
        update_profile: bool = True,
        preserve_alignment: bool = True,
    ) -> None:
        if self.image is None:
            return
        if new_profile and self.profile_update_after_id is not None:
            self.root.after_cancel(self.profile_update_after_id)
            self.profile_update_after_id = None
        if new_profile and self.roi_bounds is not None:
            self._add_profile_axis()
        first_roi = new_profile and self.roi_alignment_x is None
        if new_profile:
            self.roi_count += 1
            self.current_roi_number = self.roi_count
            if preserve_alignment and self.roi_alignment_x is not None:
                x0 = self.roi_alignment_x
        width = min(width, self.image.shape[1])
        height = min(height, self.image.shape[0])
        x0 = min(max(0, x0), self.image.shape[1] - width)
        y0 = min(max(0, y0), self.image.shape[0] - height)
        if first_roi:
            self.roi_alignment_x = x0
        x1, y1 = x0 + width, y0 + height
        self.roi_bounds = (x0, y0, width, height)
        self.roi_bounds_by_axis[self.profile_axis] = (x0, y0, width, height)
        if not new_profile:
            if self.roi_patch is not None:
                self.roi_patch.remove()
            if self.roi_label is not None:
                self.roi_label.remove()
        self.roi_patch = self.image_axis.add_patch(
            Rectangle(
                (x0, y0), width, height,
                fill=False, edgecolor="#f97316", linewidth=1.5,
            )
        )
        self.roi_label = self.image_axis.text(
            x0 + 3, y0 + 3, str(self.current_roi_number),
            color="white", fontsize=10, fontweight="bold",
            ha="left", va="top",
            bbox={"facecolor": "#f97316", "edgecolor": "none", "pad": 2},
        )
        if new_profile:
            self.roi_patches.append(self.roi_patch)
            self.roi_labels.append(self.roi_label)
        if not update_profile:
            self.canvas.draw_idle()
            return
        self._update_profile_plot()

    def _schedule_profile_update(self) -> None:
        if self.profile_update_after_id is not None:
            self.root.after_cancel(self.profile_update_after_id)
        self.profile_update_after_id = self.root.after(2000, self._apply_delayed_profile_update)

    def _apply_delayed_profile_update(self) -> None:
        self.profile_update_after_id = None
        if self.roi_bounds is not None:
            self._update_profile_plot()

    def _update_profile_plot(self) -> None:
        if self.image is None or self.roi_bounds is None:
            return
        x0, y0, width, height = self.roi_bounds
        x1, y1 = x0 + width, y0 + height
        self.profile = np.mean(self.image[y0:y1, x0:x1], axis=0)
        profile_x = np.arange(width, dtype=float)
        self.profile_data_by_axis[self.profile_axis] = (
            profile_x, self.profile.copy()
        )
        self.profile_axis.clear()
        self.profile_axis.plot(profile_x, self.profile, color="#164e63", linewidth=1.5)
        self.profile_axis.set_title(
            f"Plot profile P(x) - ROI {self.current_roi_number}: x={x0}-{x1 - 1}, y={y0}-{y1 - 1}",
            fontsize=9,
        )
        self.profile_axis.set_xlabel("Horizontal distance (pixels)")
        self.profile_axis.set_ylabel("Mean grey value")
        self.profile_axis.grid(alpha=0.2)
        self._update_profile_auc_text(self.profile_axis)
        self._align_profile_x_axes()
        self.copy_button.configure(state=tk.NORMAL)
        self.canvas.draw_idle()
        self.mode.set("Set background B(x) or select a peak")

    def connect_marker_background(self) -> None:
        marker_groups = {}
        for line, _ in self.plot_marker_lines:
            marker_groups.setdefault(line.axes, []).append(line)
        candidates = [(axis, lines) for axis, lines in marker_groups.items() if len(lines) >= 2]
        if not candidates:
            messagebox.showinfo(
                "Need two vertical lines",
                "Draw two vertical lines on the same lane plot first.",
            )
            return
        created_lines = []
        for axis, marker_lines in candidates:
            positions = sorted(line.get_xdata()[0] for line in marker_lines)
            profile_data = self.profile_data_by_axis.get(axis)
            if profile_data is None:
                continue
            profile_x, profile_y = profile_data
            values = [float(np.interp(position, profile_x, profile_y)) for position in positions]
            background_x = np.array(positions, dtype=float)
            background_y = np.array(values, dtype=float)
            for line in self.background_lines[:]:
                if line.axes is axis:
                    line.remove()
                    self.background_lines.remove(line)
            background_line, = axis.plot(
                background_x, background_y, color="#dc2626", linestyle="--",
                linewidth=1.5, marker="o", zorder=1,
            )
            self.background_lines.append(background_line)
            created_lines.append((axis, background_line))
            if axis is self.profile_axis:
                self.baseline_points = list(zip(background_x, background_y))
            self.baseline_points_by_axis[axis] = list(zip(background_x, background_y))
        self.undo_actions.append(("background_group", created_lines))
        self.canvas.draw_idle()
        self.mode.set(f"Background connected on {len(created_lines)} lane plot(s)")

    def _add_profile_axis(self) -> None:
        self.profile_axes.append(self.figure.add_axes([0.56, 0.5, 0.4, 0.4]))
        self._layout_profile_axes()
        self._align_profile_x_axes()
        self.profile_axis = self.profile_axes[-1]

    def _align_profile_x_axes(self) -> None:
        widths = [len(data[0]) for data in self.profile_data_by_axis.values()]
        if not widths:
            return
        right = max(widths) - 1
        for axis in self.profile_axes:
            axis.set_xlim(0, max(1, right))

    def _layout_profile_axes(self) -> None:
        count = len(self.profile_axes)
        row_height = 2.2
        gap = 0.35
        top_margin = 0.55
        bottom_margin = 0.65
        figure_height = max(8.0, top_margin + bottom_margin + count * row_height + (count - 1) * gap)
        self.figure.set_size_inches(11, figure_height, forward=True)
        height = row_height / figure_height
        canvas_height = int(round(figure_height * self.figure.dpi))
        self.canvas.get_tk_widget().configure(height=canvas_height)
        for index, axis in enumerate(self.profile_axes):
            y_position = (
                figure_height - top_margin - (index + 1) * row_height - index * gap
            ) / figure_height
            axis.set_position([0.52, y_position, 0.25, height])
            axis.tick_params(labelsize=8)
            if index < count - 1:
                axis.set_xlabel("")
            else:
                axis.set_xlabel("Horizontal distance (pixels)", fontsize=8)
            axis.set_ylabel("Mean grey value" if index == 0 else "", fontsize=8)
        self.marker_summary_axis.set_position(
            [0.80, bottom_margin / figure_height, 0.19,
             (figure_height - top_margin - bottom_margin) / figure_height]
        )
        self.canvas.draw_idle()
        self.root.after_idle(self._update_plot_scrollregion)

    def _update_profile_auc_text(self, axis) -> None:
        old_text = self.profile_auc_texts.get(axis)
        if old_text is not None and old_text.axes is not None:
            old_text.remove()
        auc_results = [result for result in self.auc_results if result["axis"] is axis]
        if auc_results:
            lines = [
                f"AUC{result['number']}: {result['corrected_auc']:.6g}"
                for result in auc_results
            ]
        else:
            lines = ["AUC: none"]
        self.profile_auc_texts[axis] = axis.text(
            1.02, 0.96, "\n".join(lines),
            transform=axis.transAxes, va="top", ha="left",
            fontsize=8, family="monospace", color="#166534",
            clip_on=False,
        )

    def _update_marker_summary(self) -> None:
        self.marker_summary_axis.clear()
        self.marker_summary_axis.set_title("Profile and AUC", fontsize=9)
        self.marker_summary_axis.axis("off")
        self.marker_summary_axis.text(
            0.02, 0.98,
            r"$P(x)=\frac{1}{y_2-y_1}\int_{y_1}^{y_2} I(x,y)\,dy$",
            transform=self.marker_summary_axis.transAxes,
            va="top", ha="left", fontsize=10,
        )
        self.marker_summary_axis.text(
            0.02, 0.90,
            r"$AUC=\int_{x_1}^{x_2}[P(x)-B(x)]\,dx$",
            transform=self.marker_summary_axis.transAxes,
            va="top", ha="left", fontsize=10,
        )
        lines = [
            "y = vertical pixel position",
            "x = horizontal pixel position",
            "I(x,y) = intensity at (x,y)",
            "P(x) = mean intensity across selected y range",
            "B(x) = estimated baseline between boundary points",
            "",
        ]
        for axis in self.profile_axes:
            marker_lines = [line for line, _ in self.plot_marker_lines if line.axes is axis]
            if not marker_lines:
                continue
            positions = sorted(line.get_xdata()[0] for line in marker_lines)
            title = axis.get_title().split(":")[0] or "Lane"
            lines.append(title)
            roi_bounds = self.roi_bounds_by_axis.get(axis)
            if roi_bounds is not None:
                _, y0, _, height = roi_bounds
                lines.append(f"y1 = {y0:.0f}")
                lines.append(f"y2 = {y0 + height - 1:.0f}")
                lines.append(f"height = {height:.0f} px")
            for index, position in enumerate(positions, 1):
                lines.append(f"x{index} = {position:.2f}")
            for index in range(1, len(positions)):
                difference = positions[index] - positions[index - 1]
                lines.append(f"x{index + 1}-x{index} = {difference:.2f}")
            for result in [item for item in self.auc_results if item["axis"] is axis]:
                lines.append(f"AUC{result['number']}:")
                lines.append(f"  P(x): {result['profile_auc']:.4g}")
                lines.append(f"  B(x): {result['background_auc']:.4g}")
                lines.append(f"  AUC: {result['corrected_auc']:.4g}")
            lines.append("")
        self.marker_summary_axis.text(
            0.02, 0.82, "\n".join(lines),
            transform=self.marker_summary_axis.transAxes,
            va="top", ha="left", fontsize=7, family="monospace",
        )
        for axis in self.profile_axes:
            self._update_profile_auc_text(axis)

    def enable_background(self) -> None:
        if self.profile is None:
            messagebox.showinfo("Select ROI", "Select a lane ROI first.")
            return
        self._disconnect_selectors()
        self.baseline_points = []
        self.baseline_points_by_axis[self.profile_axis] = []
        self.mode.set("Click two points on the profile to define B(x)")
        self.background_connection_id = self.canvas.mpl_connect(
            "button_press_event", self._background_click
        )

    def _background_click(self, event) -> None:
        if event.inaxes != self.profile_axis or event.xdata is None or event.ydata is None:
            return
        self.baseline_points.append((event.xdata, event.ydata))
        self.profile_axis.axvline(event.xdata, color="#d97706", alpha=0.5)
        if len(self.baseline_points) == 2:
            self._draw_baseline()
            self.mode.set("Background set. Select peaks by dragging on the profile")
        self.canvas.draw_idle()

    def _draw_baseline(self) -> None:
        if self.profile is None or len(self.baseline_points) < 2:
            return
        points = sorted(self.baseline_points)
        if points[0][0] == points[-1][0]:
            return
        x = np.arange(len(self.profile), dtype=float)
        background = np.interp(x, [point[0] for point in points], [point[1] for point in points])
        self.profile_axis.plot(x, background, color="#dc2626", linestyle="--", zorder=1)
        self.baseline_points_by_axis[self.profile_axis] = list(points)

    def _profile_for_axis(self, axis):
        return self.profile_data_by_axis.get(axis)

    def _compute_auc(self, axis, peak: tuple[float, float], show_popup: bool = True):
        profile_data = self._profile_for_axis(axis)
        baseline_points = self.baseline_points_by_axis.get(axis, self.baseline_points)
        if profile_data is None or len(baseline_points) < 2:
            if show_popup:
                messagebox.showinfo(
                    "Background required",
                    "Use Draw Background before computing AUC.",
                )
            return None
        x, profile = profile_data
        points = sorted(baseline_points)
        background = np.interp(x, [point[0] for point in points], [point[1] for point in points])
        left = max(float(x[0]), min(float(x[-1]), peak[0]))
        right = max(float(x[0]), min(float(x[-1]), peak[1]))
        if right < left:
            left, right = right, left
        left_index = max(0, int(np.floor(left)))
        right_index = min(len(x) - 1, int(np.ceil(right)))
        if right_index <= left_index:
            return None
        area_x = x[left_index : right_index + 1]
        area_profile = profile[left_index : right_index + 1]
        area_background = background[left_index : right_index + 1]
        area_fill = axis.fill_between(
                area_x, area_profile, area_background,
            color="#16a34a", alpha=0.25, label="_nolegend_", zorder=0,
            )
        self.auc_artists.append(area_fill)
        axis.axvline(area_x[0], color="#16a34a", alpha=0.7)
        axis.axvline(area_x[-1], color="#16a34a", alpha=0.7)
        profile_auc = float(np.trapezoid(area_profile, area_x))
        background_auc = float(np.trapezoid(area_background, area_x))
        corrected_auc = profile_auc - background_auc
        existing_result = next(
            (
                result for result in self.auc_results
                if result["axis"] is axis
                and result["start"] == area_x[0]
                and result["end"] == area_x[-1]
            ),
            None,
        )
        self.auc_results = [
            result for result in self.auc_results
            if not (
                result["axis"] is axis
                and result["start"] == area_x[0]
                and result["end"] == area_x[-1]
            )
        ]
        auc_number = (
            existing_result["number"]
            if existing_result is not None
            else max((result["number"] for result in self.auc_results), default=0) + 1
        )
        self.auc_results.append({
            "axis": axis,
            "number": auc_number,
            "start": float(area_x[0]),
            "end": float(area_x[-1]),
            "profile_auc": profile_auc,
            "background_auc": background_auc,
            "corrected_auc": corrected_auc,
        })
        area_label = axis.text(
            (area_x[0] + area_x[-1]) / 2,
            max(0.55, 0.92 - 0.1 * ((auc_number - 1) % 4)),
            f"AUC{auc_number}",
            transform=axis.get_xaxis_transform(),
            ha="center",
            va="top",
            color="#166534",
            fontsize=9,
            fontweight="bold",
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "#16a34a"},
        )
        self.auc_artists.append(area_label)
        self._update_marker_summary()
        self.mode.set(f"Integrated density: {corrected_auc:.3g} mean-grey-value pixels")
        if show_popup:
            messagebox.showinfo(
                "Background-subtracted AUC",
                f"Enclosed region: x={area_x[0]:.1f} to x={area_x[-1]:.1f}\n"
                f"Total integrated density: {corrected_auc:.6g} mean-grey-value pixels",
            )
        return profile_auc, background_auc, corrected_auc

    def _draw_corrected_auc(self, peak: tuple[float, float]) -> None:
        self._compute_auc(self.profile_axis, peak, show_popup=False)

    def enable_wand(self) -> None:
        if self.profile is None:
            messagebox.showinfo("Select ROI", "Select a lane ROI first.")
            return
        self._disconnect_selectors()
        self.mode.set("Click inside the closed green shape on a lane plot")
        self.auc_click_connection_id = self.canvas.mpl_connect(
            "button_press_event", self._auc_click
        )

    def _compute_all_auc_regions(self) -> None:
        marker_groups = {}
        for line, _ in self.plot_marker_lines:
            marker_groups.setdefault(line.axes, []).append(line)
        results = []
        for axis, marker_lines in marker_groups.items():
            if len(marker_lines) < 2:
                continue
            positions = sorted(line.get_xdata()[0] for line in marker_lines)
            for index, bounds in enumerate(zip(positions[:-1], positions[1:]), 1):
                values = self._compute_auc(axis, bounds, show_popup=False)
                if values is not None:
                    profile_auc, background_auc, corrected_auc = values
                    results.append(
                        f"{axis.get_title().split(':')[0]} interval {index} "
                        f"({bounds[0]:.1f}-{bounds[1]:.1f})\n"
                        f"  P(x) integrated density: {profile_auc:.6g}\n"
                        f"  Background integrated density: {background_auc:.6g}\n"
                        f"  Total integrated density: {corrected_auc:.6g}"
                    )
        self.canvas.draw_idle()
        if not results:
            messagebox.showinfo(
                "AUC requires enclosed regions",
                "Draw vertical x lines and connect the background before computing AUC.",
            )
            return
        self.mode.set(f"Computed {len(results)} enclosed AUC region(s)")
        self._show_scrollable_auc_results(
            "\n".join(results)
            + "\n\nOnly the area between each consecutive x line, P(x), and the connected background was measured."
        )

    def _show_scrollable_auc_results(self, content: str) -> None:
        window = tk.Toplevel(self.root)
        window.title("Background-subtracted AUC")
        window.geometry("620x500")
        window.transient(self.root)

        text_frame = ttk.Frame(window, padding=8)
        text_frame.pack(fill=tk.BOTH, expand=True)
        scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL)
        output = tk.Text(
            text_frame, wrap=tk.WORD, yscrollcommand=scrollbar.set,
            state=tk.NORMAL, padx=8, pady=8,
        )
        scrollbar.configure(command=output.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        output.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        output.insert("1.0", content)
        output.configure(state=tk.DISABLED)
        ttk.Button(window, text="Close", command=window.destroy).pack(pady=(0, 8))

    def _auc_click(self, event) -> None:
        if event.inaxes not in self.profile_axes or event.xdata is None:
            return
        marker_lines = [line for line, _ in self.plot_marker_lines if line.axes is event.inaxes]
        if len(marker_lines) < 2:
            messagebox.showinfo(
                "Need two vertical lines",
                "Draw at least two vertical lines on this lane plot first.",
            )
            return
        if not any(line.axes is event.inaxes for line in self.background_lines):
            messagebox.showinfo(
                "Background required",
                "Connect the background line on this lane plot before using the wand.",
            )
            return
        positions = sorted(line.get_xdata()[0] for line in marker_lines)
        click_position = event.xdata
        pair = next(
            (
                bounds for bounds in zip(positions[:-1], positions[1:])
                if bounds[0] <= click_position <= bounds[1]
            ),
            None,
        )
        if pair is None:
            messagebox.showinfo(
                "Outside enclosed region",
                "Click between two consecutive vertical x lines.",
            )
            return
        profile_data = self.profile_data_by_axis.get(event.inaxes)
        baseline_points = self.baseline_points_by_axis.get(event.inaxes, [])
        if profile_data is None or len(baseline_points) < 2:
            return
        profile_x, profile_y = profile_data
        background_y = np.interp(
            click_position,
            [point[0] for point in sorted(baseline_points)],
            [point[1] for point in sorted(baseline_points)],
        )
        profile_y_at_click = float(np.interp(click_position, profile_x, profile_y))
        if not min(profile_y_at_click, background_y) <= event.ydata <= max(profile_y_at_click, background_y):
            messagebox.showinfo(
                "Outside enclosed shape",
                "Click between P(x) and the connected background line.",
            )
            return
        self._compute_auc(event.inaxes, pair, show_popup=False)
        self.canvas.draw_idle()

    def _peak_selected(self, xmin: float, xmax: float) -> None:
        if xmax - xmin < 1:
            return
        peak = tuple(sorted((xmin, xmax)))
        self.peaks.append(peak)
        peak_artist = self.profile_axis.axvspan(*peak, color="#16a34a", alpha=0.2)
        self.undo_actions.append(("peak", peak_artist, peak))
        self._draw_corrected_auc(peak)
        self.canvas.draw_idle()

    def undo(self) -> None:
        if not self.undo_actions:
            self.mode.set("Nothing to undo")
            return
        action = self.undo_actions.pop()
        if action[0] == "marker_group":
            marker_group = action[1]
            for _, line, marker_text in marker_group:
                line.remove()
                marker_text.remove()
                marker_pair = (line, marker_text)
                if marker_pair in self.plot_marker_lines:
                    self.plot_marker_lines.remove(marker_pair)
            for line, marker_text in action[2]:
                line.remove()
                marker_pair = (line, marker_text)
                if marker_pair in self.image_marker_lines:
                    self.image_marker_lines.remove(marker_pair)
            if self.plot_marker_positions:
                self.plot_marker_positions.pop()
            self._update_marker_summary()
        elif action[0] == "background":
            target_axis, background_line = action[1], action[2]
            background_line.remove()
            if background_line in self.background_lines:
                self.background_lines.remove(background_line)
        elif action[0] == "background_group":
            for target_axis, background_line in action[1]:
                background_line.remove()
                if background_line in self.background_lines:
                    self.background_lines.remove(background_line)
        elif action[0] == "peak":
            peak_artist, peak = action[1], action[2]
            peak_artist.remove()
            if self.peaks and self.peaks[-1] == peak:
                self.peaks.pop()
        self.canvas.draw_idle()
        self.mode.set("Last action undone")

    def _disconnect_selectors(self) -> None:
        if self.roi_selector is not None:
            self.roi_selector.set_active(False)
        if self.crop_selector is not None:
            self.crop_selector.set_active(False)
            self.crop_selector.set_visible(False)
        if self.span_selector is not None:
            self.span_selector.set_active(False)
        if self.background_connection_id is not None:
            self.canvas.mpl_disconnect(self.background_connection_id)
            self.background_connection_id = None
        if self.copy_connection_id is not None:
            self.canvas.mpl_disconnect(self.copy_connection_id)
            self.copy_connection_id = None
        if self.vertical_line_connection_id is not None:
            self.canvas.mpl_disconnect(self.vertical_line_connection_id)
            self.vertical_line_connection_id = None
        if self.auc_click_connection_id is not None:
            self.canvas.mpl_disconnect(self.auc_click_connection_id)
            self.auc_click_connection_id = None

    def _update_display(self, _value=None) -> None:
        if self.image_artist is None or self.image is None:
            return
        data_min, data_max = self._display_reference_range()
        midpoint = (data_min + data_max) / 2 + self.brightness.get() * (data_max - data_min) / 2
        span = max((data_max - data_min) / self.contrast.get(), 1e-12)
        self.image_artist.set_clim(midpoint - span / 2, midpoint + span / 2)
        self.canvas.draw_idle()

    def _display_reference_range(self) -> tuple[float, float]:
        if self.image is None:
            return 0.0, 1.0
        data_min, data_max = np.percentile(self.image, (0.35, 99.65))
        if data_min >= data_max:
            data_min, data_max = float(np.min(self.image)), float(np.max(self.image))
        return float(data_min), float(data_max)

    def reset_display_range(self) -> None:
        self.brightness.set(0.0)
        self.contrast.set(1.0)
        self._update_display()
        self.mode.set("ImageJ-style auto display range restored")

    def reset(self, clear_image: bool = True) -> None:
        if self.profile_update_after_id is not None:
            self.root.after_cancel(self.profile_update_after_id)
            self.profile_update_after_id = None
        self._disconnect_selectors()
        self.profile = None
        self.roi_bounds = None
        if self.roi_patch is not None:
            self.roi_patch.remove()
            self.roi_patch = None
        if self.roi_label is not None:
            self.roi_label.remove()
            self.roi_label = None
        for patch in self.roi_patches:
            if patch in self.image_axis.patches:
                patch.remove()
        for label in self.roi_labels:
            if label in self.image_axis.texts:
                label.remove()
        self.roi_patches = []
        self.roi_labels = []
        self.roi_count = 0
        self.current_roi_number = 0
        self.roi_alignment_x = None
        self.baseline_points = []
        self.peaks = []
        self.auc_artists = []
        self.auc_results = []
        self.profile_auc_texts = {}
        self.plot_marker_lines = []
        self.image_marker_lines = []
        self.plot_marker_positions = []
        self.undo_actions = []
        self.profile_data_by_axis = {}
        self.roi_bounds_by_axis = {}
        self.background_lines = []
        self.baseline_points_by_axis = {}
        self.marker_summary_axis.clear()
        self.marker_summary_axis.set_title("Profile and AUC", fontsize=9)
        self.marker_summary_axis.axis("off")
        for axis in self.profile_axes[1:]:
            axis.remove()
        self.profile_axes = self.profile_axes[:1]
        self.profile_axis = self.profile_axes[0]
        self._layout_profile_axes()
        self.profile_axis.clear()
        self.profile_axis.set_title("Plot profile P(x)")
        self.profile_axis.set_xlabel("Horizontal distance (pixels)")
        self.profile_axis.set_ylabel("Mean grey value")
        if clear_image:
            self.image = None
            self.original_image = None
            self.image_artist = None
            self.roi_patch = None
            self.image_axis.clear()
            self.image_axis.set_title("Open a grayscale TIFF")
            self.copy_button.configure(state=tk.DISABLED)
        self.canvas.draw_idle()
        self.mode.set("Open a TIFF to begin" if clear_image else "Select a horizontal lane ROI on the image")

    def export_csv(self) -> None:
        profile_items = [
            (axis, self.profile_data_by_axis.get(axis))
            for axis in self.profile_axes
            if self.profile_data_by_axis.get(axis) is not None
        ][:10]
        if not profile_items:
            messagebox.showinfo("Nothing to export", "Select at least one lane ROI first.")
            return
        path = filedialog.asksaveasfilename(
            title="Export plot profiles",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
        )
        if not path:
            return
        profiles = []
        header = ["x distance"]
        for index, (_, profile_data) in enumerate(profile_items, 1):
            profile_x, profile_y = profile_data
            header.append(f"ROI {index} profile")
            profiles.append((profile_x, profile_y))

        csv_rows = []
        shared_x = profiles[0][0]
        for row_index in range(max(len(profile_x) for profile_x, _ in profiles)):
            row = [shared_x[row_index] if row_index < len(shared_x) else ""]
            for _, profile_y in profiles:
                if row_index < len(profile_y):
                    row.append(profile_y[row_index])
                else:
                    row.append("")
            csv_rows.append(row)
        with open(path, "w", newline="", encoding="utf-8") as output:
            writer = csv.writer(output)
            writer.writerow(header)
            writer.writerows(csv_rows)
        self.mode.set(f"Exported {len(profile_items)} lane profile(s)")

    def export_auc_csv(self) -> None:
        if not self.auc_results:
            messagebox.showinfo("Nothing to export", "Use Magic Wand to calculate at least one AUC first.")
            return
        path = filedialog.asksaveasfilename(
            title="Export all AUC results",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
        )
        if not path:
            return
        with open(path, "w", newline="", encoding="utf-8") as output:
            writer = csv.writer(output)
            writer.writerow([
                "AUC", "ROI", "profile", "start_x", "end_x",
                "profile_auc", "background_auc", "corrected_auc",
            ])
            for result in self.auc_results:
                writer.writerow([
                    f"AUC{result['number']}",
                    self.profile_axes.index(result["axis"]) + 1,
                    result["axis"].get_title().split(":")[0],
                    result["start"],
                    result["end"],
                    result["profile_auc"],
                    result["background_auc"],
                    result["corrected_auc"],
                ])
        self.mode.set(f"Exported {len(self.auc_results)} AUC result(s)")
        messagebox.showinfo(
            "Export complete",
            f"Saved {len(self.auc_results)} result(s) to {Path(path).name}.",
        )


def main() -> None:
    root = tk.Tk()
    DensitometryApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
