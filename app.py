from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PySide6.QtCore import QThread, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFrame,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from dino_backend import DEFAULT_MODEL, DinoFeatureExtractor


ROOT_DIR = Path(__file__).resolve().parent
TEST_IMAGES_DIR = ROOT_DIR / "test_images"
VALID_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def list_test_images() -> list[str]:
    if not TEST_IMAGES_DIR.exists():
        return []

    return sorted(
        path.name
        for path in TEST_IMAGES_DIR.iterdir()
        if path.is_file()
        and not path.name.startswith(".")
        and path.suffix.lower() in VALID_SUFFIXES
    )


def default_pair(image_names: list[str]) -> tuple[str | None, str | None]:
    if not image_names:
        return None, None
    if len(image_names) == 1:
        return image_names[0], image_names[0]
    return image_names[0], image_names[1]


@dataclass
class SimilarityResult:
    request_id: int
    patch_x: int
    patch_y: int
    grid_width: int
    grid_height: int
    raw_min: float
    raw_max: float
    heatmap: torch.Tensor
    output_width: int
    output_height: int


class SimilarityWorker(QThread):
    completed = Signal(object)
    failed = Signal(int, str)

    def __init__(
        self,
        *,
        request_id: int,
        extractor: DinoFeatureExtractor,
        left_path: Path,
        right_path: Path,
        x_norm: float,
        y_norm: float,
    ):
        super().__init__()
        self.request_id = request_id
        self.extractor = extractor
        self.left_path = left_path
        self.right_path = right_path
        self.x_norm = x_norm
        self.y_norm = y_norm

    def run(self):
        try:
            left_features = self.extractor.dense_features(self.left_path)
            right_features = self.extractor.dense_features(self.right_path)
            patch_y, patch_x, raw_similarity, heatmap = compute_similarity_map(
                left_features.patch_grid,
                right_features.patch_grid,
                self.x_norm,
                self.y_norm,
            )
            self.completed.emit(
                SimilarityResult(
                    request_id=self.request_id,
                    patch_x=patch_x,
                    patch_y=patch_y,
                    grid_width=left_features.patch_grid.shape[1],
                    grid_height=left_features.patch_grid.shape[0],
                    raw_min=float(raw_similarity.min().item()),
                    raw_max=float(raw_similarity.max().item()),
                    heatmap=heatmap,
                    output_width=right_features.original_width,
                    output_height=right_features.original_height,
                )
            )
        except Exception as exc:
            self.failed.emit(self.request_id, str(exc))


class ImageView(QGraphicsView):
    image_clicked = Signal(float, float)

    def __init__(self):
        super().__init__()
        self._scene = QGraphicsScene(self)
        self._pixmap_item = QGraphicsPixmapItem()
        self._overlay_item = QGraphicsPixmapItem()
        self._selection_item = QGraphicsRectItem()
        self._placeholder = self._scene.addText("No image selected")
        self._scene.addItem(self._pixmap_item)
        self._scene.addItem(self._overlay_item)
        self._scene.addItem(self._selection_item)
        self.setScene(self._scene)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setRenderHints(
            QPainter.RenderHint.Antialiasing
            | QPainter.RenderHint.SmoothPixmapTransform
        )
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setBackgroundBrush(QColor("#f4f4f1"))
        self.setMinimumHeight(520)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._overlay_item.setOpacity(0.56)
        self._overlay_item.setZValue(1)
        self._selection_item.setZValue(2)
        self._selection_item.setPen(QPen(QColor("#f97316"), 3))
        self._selection_item.hide()

    def set_image_path(self, image_path: Path | None):
        self.clear_overlay()
        self.clear_selection()
        if image_path is None or not image_path.exists():
            self._pixmap_item.setPixmap(QPixmap())
            self._placeholder.setVisible(True)
            self._scene.setSceneRect(self._placeholder.sceneBoundingRect())
            return

        pixmap = QPixmap(str(image_path))
        if pixmap.isNull():
            self._pixmap_item.setPixmap(QPixmap())
            self._placeholder.setPlainText(f"Failed to load\n{image_path.name}")
            self._placeholder.setVisible(True)
            self._scene.setSceneRect(self._placeholder.sceneBoundingRect())
            return

        self._placeholder.setVisible(False)
        self._pixmap_item.setPixmap(pixmap)
        self._overlay_item.setOffset(0, 0)
        self._scene.setSceneRect(self._pixmap_item.boundingRect())
        self.fitInView(self._pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)

    def set_overlay_pixmap(self, pixmap: QPixmap | None):
        if pixmap is None or pixmap.isNull():
            self.clear_overlay()
            return

        self._overlay_item.setPixmap(pixmap)
        self._overlay_item.setOffset(0, 0)
        self._overlay_item.show()

    def clear_overlay(self):
        self._overlay_item.setPixmap(QPixmap())
        self._overlay_item.hide()

    def set_selection_patch(self, patch_x: int, patch_y: int, grid_w: int, grid_h: int):
        pixmap = self._pixmap_item.pixmap()
        if pixmap.isNull() or grid_w <= 0 or grid_h <= 0:
            self.clear_selection()
            return

        patch_width = pixmap.width() / grid_w
        patch_height = pixmap.height() / grid_h
        self._selection_item.setRect(
            patch_x * patch_width,
            patch_y * patch_height,
            patch_width,
            patch_height,
        )
        self._selection_item.show()

    def clear_selection(self):
        self._selection_item.hide()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if not self._pixmap_item.pixmap().isNull():
            self.fitInView(self._pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            point = self.mapToScene(event.position().toPoint())
            rect = self._pixmap_item.sceneBoundingRect()
            if rect.contains(point) and rect.width() > 0 and rect.height() > 0:
                x_norm = (point.x() - rect.left()) / rect.width()
                y_norm = (point.y() - rect.top()) / rect.height()
                self.image_clicked.emit(float(x_norm), float(y_norm))
        super().mousePressEvent(event)


class ImagePane(QWidget):
    def __init__(self, title: str):
        super().__init__()
        self.title_label = QLabel(title)
        self.title_label.setStyleSheet("font-size: 18px; font-weight: 600;")

        self.dropdown = QComboBox()
        self.dropdown.setMinimumWidth(260)

        self.view = ImageView()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        layout.addWidget(self.title_label)
        layout.addWidget(self.dropdown)
        layout.addWidget(self.view, 1)

        self.dropdown.currentTextChanged.connect(self._update_image)

    def set_choices(self, image_names: list[str], selected_name: str | None):
        self.dropdown.blockSignals(True)
        self.dropdown.clear()
        self.dropdown.addItems(image_names)

        if image_names:
            index = (
                image_names.index(selected_name)
                if selected_name in image_names
                else 0
            )
            self.dropdown.setCurrentIndex(index)
        self.dropdown.blockSignals(False)
        self._update_image(self.dropdown.currentText())

    def current_name(self) -> str | None:
        value = self.dropdown.currentText().strip()
        return value or None

    def _update_image(self, selection: str):
        image_path = TEST_IMAGES_DIR / selection if selection else None
        self.view.set_image_path(image_path)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Semantic Similarity Explorer")
        self.resize(1400, 900)
        self.extractor = DinoFeatureExtractor(model_name=DEFAULT_MODEL)
        self._active_workers: set[SimilarityWorker] = set()
        self._latest_request_id = 0

        container = QWidget()
        self.setCentralWidget(container)

        root_layout = QVBoxLayout(container)
        root_layout.setContentsMargins(20, 20, 20, 20)
        root_layout.setSpacing(16)

        title = QLabel("Semantic Similarity Explorer")
        title.setStyleSheet("font-size: 28px; font-weight: 700;")

        toolbar = QHBoxLayout()
        toolbar.setSpacing(12)
        self.refresh_button = QPushButton("Refresh image list")
        self.refresh_button.clicked.connect(self.refresh_images)
        toolbar.addStretch(1)
        toolbar.addWidget(self.refresh_button, 0)

        panes_layout = QHBoxLayout()
        panes_layout.setSpacing(18)
        self.left_pane = ImagePane("Left image")
        self.right_pane = ImagePane("Right image")
        panes_layout.addWidget(self.left_pane, 1)
        panes_layout.addWidget(self.right_pane, 1)

        root_layout.addWidget(title)
        root_layout.addLayout(toolbar)
        root_layout.addLayout(panes_layout, 1)

        self.left_pane.dropdown.currentTextChanged.connect(self._on_selection_changed)
        self.right_pane.dropdown.currentTextChanged.connect(self._on_selection_changed)
        self.left_pane.view.image_clicked.connect(self.update_similarity_from_click)

        self.refresh_images()

    def refresh_images(self):
        image_names = list_test_images()
        default_left, default_right = default_pair(image_names)

        left_selected = self.left_pane.current_name() or default_left
        right_selected = self.right_pane.current_name() or default_right

        self.left_pane.set_choices(image_names, left_selected)
        self.right_pane.set_choices(image_names, right_selected)
        self.right_pane.view.clear_overlay()
        self.left_pane.view.clear_selection()

    def _on_selection_changed(self):
        self._latest_request_id += 1
        self.right_pane.view.clear_overlay()
        self.left_pane.view.clear_selection()

    def update_similarity_from_click(self, x_norm: float, y_norm: float):
        left_name = self.left_pane.current_name()
        right_name = self.right_pane.current_name()
        if not left_name or not right_name:
            return

        left_path = TEST_IMAGES_DIR / left_name
        right_path = TEST_IMAGES_DIR / right_name

        self._latest_request_id += 1
        request_id = self._latest_request_id
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        QApplication.processEvents()
        worker = SimilarityWorker(
            request_id=request_id,
            extractor=self.extractor,
            left_path=left_path,
            right_path=right_path,
            x_norm=x_norm,
            y_norm=y_norm,
        )
        worker.completed.connect(self._handle_similarity_result)
        worker.failed.connect(self._handle_similarity_error)
        worker.finished.connect(lambda: self._cleanup_worker(worker))
        self._active_workers.add(worker)
        worker.start()

    def _cleanup_worker(self, worker: SimilarityWorker):
        self._active_workers.discard(worker)
        if not self._active_workers:
            QApplication.restoreOverrideCursor()

    def _handle_similarity_result(self, result: SimilarityResult):
        if result.request_id != self._latest_request_id:
            return

        overlay = make_heatmap_pixmap(
            result.heatmap,
            width=result.output_width,
            height=result.output_height,
        )
        self.left_pane.view.set_selection_patch(
            patch_x=result.patch_x,
            patch_y=result.patch_y,
            grid_w=result.grid_width,
            grid_h=result.grid_height,
        )
        self.right_pane.view.set_overlay_pixmap(overlay)

    def _handle_similarity_error(self, request_id: int, message: str):
        if request_id != self._latest_request_id:
            return

        QMessageBox.critical(self, "Similarity Error", message)


def compute_similarity_map(
    left_grid: torch.Tensor,
    right_grid: torch.Tensor,
    x_norm: float,
    y_norm: float,
) -> tuple[int, int, torch.Tensor, torch.Tensor]:
    left_height, left_width, _ = left_grid.shape
    patch_x = min(max(int(x_norm * left_width), 0), left_width - 1)
    patch_y = min(max(int(y_norm * left_height), 0), left_height - 1)
    query = left_grid[patch_y, patch_x]
    raw_similarity = torch.einsum("hwd,d->hw", right_grid, query)
    min_value = raw_similarity.min()
    max_value = raw_similarity.max()
    if float(max_value - min_value) < 1e-8:
        normalized = torch.zeros_like(raw_similarity)
    else:
        normalized = (raw_similarity - min_value) / (max_value - min_value)
    return patch_y, patch_x, raw_similarity, normalized


def make_heatmap_pixmap(heatmap: torch.Tensor, width: int, height: int) -> QPixmap:
    heatmap = heatmap.to(dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    upsampled = F.interpolate(
        heatmap,
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )[0, 0].clamp(0.0, 1.0)

    four_x = 4.0 * upsampled
    red = torch.clamp(torch.minimum(four_x - 1.5, -four_x + 4.5), 0.0, 1.0)
    green = torch.clamp(torch.minimum(four_x - 0.5, -four_x + 3.5), 0.0, 1.0)
    blue = torch.clamp(torch.minimum(four_x + 0.5, -four_x + 2.5), 0.0, 1.0)
    alpha = upsampled * 0.78

    rgba = torch.stack([red, green, blue, alpha], dim=-1)
    rgba = (rgba * 255).to(torch.uint8).cpu().numpy()
    image = np.ascontiguousarray(rgba)
    qimage = QImage(
        image.data,
        width,
        height,
        4 * width,
        QImage.Format.Format_RGBA8888,
    ).copy()
    return QPixmap.fromImage(qimage)


def build_window() -> MainWindow:
    return MainWindow()


def main():
    app = QApplication.instance() or QApplication(sys.argv)
    window = build_window()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
