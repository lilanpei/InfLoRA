"""Whitened Nearest Class Mean (W-NCM) classifier.

Implements a shrinkage-whitened cosine nearest-class-mean classifier using
Welford's algorithm for both global and class statistics (no EMA), with
unified GPU/CPU support.
"""

import numpy as np
import logging
from scipy.linalg import eigh

# Configure logger
logger = logging.getLogger(__name__)

# Try to import torch for GPU support
try:
    import torch

    TORCH_AVAILABLE = True
    GPU_AVAILABLE = torch.cuda.is_available()
except ImportError:  # pragma: no cover - torch may not be installed
    TORCH_AVAILABLE = False
    GPU_AVAILABLE = False


class WhitenedNCMClassifier:
    """Whitened Nearest Class Mean (W-NCM) classifier (unified GPU/CPU).

    Uses Welford's algorithm for:
    - Global covariance Σ (exact)
    - Class prototypes µ_c (exact, no EMA)

    Minimal hyperparameters needed.
    """

    def __init__(
        self,
        feature_dim: int,
        shrinkage_lambda: float = 0.01,
        device: str = "auto",  # auto-detect by default
        use_whitening: bool = True,
    ):
        """Initialize Whitened NCM (W-NCM) classifier.

        Args:
            feature_dim: Dimensionality of input features
            shrinkage_lambda: Shrinkage parameter (λ)
            device: Device for computation ("auto", "cuda", or "cpu")
            use_whitening: Whether to apply whitening (if False, falls back to
                cosine on raw features)
        """
        self.feature_dim = feature_dim
        self.shrinkage_lambda = shrinkage_lambda
        self.use_whitening = use_whitening

        # Auto-detect device
        if device == "auto":
            if TORCH_AVAILABLE and GPU_AVAILABLE:
                self.device = "cuda"
                self.use_gpu = True
                logger.info("WhitenedNCM: Using GPU acceleration")
            else:
                self.device = "cpu"
                self.use_gpu = False
                if TORCH_AVAILABLE:
                    logger.info("WhitenedNCM: GPU not available, using CPU")
                else:
                    logger.info("WhitenedNCM: PyTorch not installed, using CPU")
        else:
            self.device = device
            self.use_gpu = device == "cuda" and TORCH_AVAILABLE and GPU_AVAILABLE
            if self.use_gpu:
                logger.info("WhitenedNCM: Using GPU acceleration")
            else:
                logger.info("WhitenedNCM: Using CPU")

        # Initialize data structures based on device
        if self.use_gpu:
            # GPU data structures
            self.global_mean = torch.zeros(feature_dim, device="cuda")
            self.global_cov = torch.eye(feature_dim, device="cuda")
            self.sum_sq_centered = torch.zeros(
                (feature_dim, feature_dim), device="cuda", dtype=torch.float64
            )
            self.n_samples = 0

            # Per-class statistics (GPU) - memory efficient: only store means and counts
            self.class_stats = {}  # {class_id: {"mean": ..., "n": ...}}

            # Whitening transform (GPU)
            self.whitening_matrix = None

        else:
            # CPU data structures (numpy)
            self.global_mean = np.zeros(feature_dim, dtype=np.float32)
            self.global_cov = np.eye(feature_dim, dtype=np.float32)
            self.sum_sq_centered = np.zeros(
                (feature_dim, feature_dim), dtype=np.float64
            )
            self.n_samples = 0

            # Per-class statistics (CPU) - memory efficient: only store means and counts
            self.class_stats = {}  # {class_id: {"mean": ..., "n": ...}}

            # Whitening transform (CPU)
            self.whitening_matrix = None

        # Bookkeeping (same for both)
        self.seen_classes = []

    def _to_device(self, data):
        """Convert data to appropriate device format."""
        if self.use_gpu:
            if isinstance(data, np.ndarray):
                # Avoid gradient tracking for inference tensors
                tensor = torch.from_numpy(data).float().cuda()
                if tensor.requires_grad:
                    tensor = tensor.detach()
                return tensor
            elif hasattr(data, "detach") and data.requires_grad:
                return data.detach()
            return data
        else:
            if hasattr(data, "cpu"):
                return data.cpu().numpy()
            return data

    def _update_global_statistics(self, X):
        """Update global mean and covariance via Welford's algorithm."""
        X = self._to_device(X)

        n_new = len(X)
        n_old = self.n_samples
        n_total = n_old + n_new

        if n_old == 0:
            # Initialize
            if self.use_gpu:
                self.global_mean = X.mean(dim=0)
                centered = X - self.global_mean
                self.sum_sq_centered = (centered.T @ centered).double()
            else:
                self.global_mean = X.mean(axis=0).astype(np.float32)
                centered = X - self.global_mean
                self.sum_sq_centered = (centered.T @ centered).astype(np.float64)
        else:
            # Batched Welford's algorithm (O(D²) instead of O(N·D²))
            if self.use_gpu:
                old_mean = self.global_mean.clone()
                self.global_mean = (old_mean * n_old + X.sum(dim=0)) / n_total

                # Batched update: sum of outer products via matrix operations
                delta_old = X - old_mean  # N x D
                delta_new = X - self.global_mean  # N x D
                # sum of outer products = delta_old.T @ delta_new
                self.sum_sq_centered += (delta_old.T @ delta_new).double()
            else:
                old_mean = self.global_mean.copy()
                self.global_mean = (old_mean * n_old + X.sum(axis=0)) / n_total

                # Batched update: sum of outer products via matrix operations
                delta_old = X - old_mean  # N x D
                delta_new = X - self.global_mean  # N x D
                # sum of outer products = delta_old.T @ delta_new
                self.sum_sq_centered += delta_old.T @ delta_new

        self.n_samples = n_total

        # Compute covariance matrix
        if self.n_samples > 1:
            if self.use_gpu:
                self.global_cov = (self.sum_sq_centered / (self.n_samples - 1)).float()
            else:
                self.global_cov = (self.sum_sq_centered / (self.n_samples - 1)).astype(
                    np.float32
                )
        else:
            if self.use_gpu:
                self.global_cov = torch.eye(self.feature_dim, device="cuda")
            else:
                self.global_cov = np.eye(self.feature_dim, dtype=np.float32)

    def _update_class_statistics_welford(self, X, y):
        """Update per-class statistics via Welford's algorithm (NO EMA!)."""
        X = self._to_device(X)
        y = self._to_device(y)  # Convert labels to same device

        unique_classes = torch.unique(y) if self.use_gpu else np.unique(y)

        for class_id in unique_classes:
            if self.use_gpu:
                class_mask = y == class_id
                class_X = X[class_mask]
            else:
                class_mask = y == class_id
                class_X = X[class_mask]

            n_new = len(class_X)

            class_id_int = int(class_id.cpu().item() if self.use_gpu else class_id)
            if class_id_int not in self.class_stats:
                # Initialize class statistics - memory efficient: only mean and count
                if self.use_gpu:
                    self.class_stats[class_id_int] = {
                        "mean": class_X.mean(dim=0),
                        "n": n_new,
                    }
                else:
                    self.class_stats[class_id_int] = {
                        "mean": class_X.mean(axis=0),
                        "n": n_new,
                    }
                self.seen_classes.append(class_id_int)
            else:
                # Update class mean (exact incremental average)
                stats = self.class_stats[class_id_int]
                n_old = stats["n"]
                old_mean = (
                    stats["mean"].clone() if self.use_gpu else stats["mean"].copy()
                )
                n_total = n_old + n_new

                # Update mean exactly
                if self.use_gpu:
                    new_mean = (old_mean * n_old + class_X.sum(dim=0)) / n_total
                    stats["mean"] = new_mean
                else:
                    new_mean = (old_mean * n_old + class_X.sum(axis=0)) / n_total
                    stats["mean"] = new_mean

                stats["n"] = n_total

    def _apply_shrinkage(self):
        """Apply shrinkage: Σ ← (1−λ)Σ + λdiag(Σ)."""
        if not getattr(self, "use_whitening", True):
            return
        if self.shrinkage_lambda > 0:
            if self.use_gpu:
                diag_cov = torch.diag(torch.diag(self.global_cov))
                self.global_cov = (1.0 - self.shrinkage_lambda) * self.global_cov + (
                    self.shrinkage_lambda * diag_cov
                )
            else:
                diag_cov = np.diag(np.diag(self.global_cov))
                self.global_cov = (1.0 - self.shrinkage_lambda) * self.global_cov + (
                    self.shrinkage_lambda * diag_cov
                )
        # Invalidate cached whitening matrix
        self.whitening_matrix = None

    def _compute_whitening_matrix(self):
        """Compute whitening matrix W = Σ^(-1/2)."""
        if self.whitening_matrix is not None:
            return self.whitening_matrix

        try:
            if self.use_gpu:
                # GPU eigenvalue decomposition
                eigenvals, eigenvecs = torch.linalg.eigh(self.global_cov)
                eigenvals = torch.clamp(eigenvals, min=1e-12)  # Numerical stability

                # Whitening matrix
                whitening = (
                    eigenvecs @ torch.diag(1.0 / torch.sqrt(eigenvals)) @ eigenvecs.T
                )
                self.whitening_matrix = whitening.float()
                return self.whitening_matrix
            else:
                # CPU eigenvalue decomposition
                eigenvals, eigenvecs = eigh(self.global_cov)
                eigenvals = np.maximum(eigenvals, 1e-12)  # Numerical stability

                # Whitening matrix
                whitening = eigenvecs @ np.diag(1.0 / np.sqrt(eigenvals)) @ eigenvecs.T
                self.whitening_matrix = whitening.astype(np.float32)
                return self.whitening_matrix
        except Exception:  # pragma: no cover - numerical fallback
            if self.use_gpu:
                self.whitening_matrix = torch.eye(self.feature_dim, device="cuda")
            else:
                self.whitening_matrix = np.eye(self.feature_dim, dtype=np.float32)
            return self.whitening_matrix

    def _whiten_features(self, X):
        """Apply whitening transformation."""
        X = self._to_device(X)
        if not getattr(self, "use_whitening", True):
            return X
        if self.whitening_matrix is None:
            self._compute_whitening_matrix()
        if self.use_gpu:
            return (X - self.global_mean) @ self.whitening_matrix.T
        else:
            return (X - self.global_mean) @ self.whitening_matrix

    def update(self, features: np.ndarray, labels: np.ndarray, _task_id: int):
        """Update W-NCM statistics with a new batch of features and labels.

        Mirrors the original inline implementation used in DC-LoRA's training
        script: update global covariance via Welford, apply shrinkage, then
        update per-class means via Welford.
        """
        # Step 1: Update Σ via Welford's algorithm
        self._update_global_statistics(features)

        # Step 2: Apply shrinkage
        self._apply_shrinkage()

        # Step 3: Update class statistics via Welford's algorithm (NO EMA!)
        self._update_class_statistics_welford(features, labels)

    def post_update_hook(self, _task_id: int):
        """Post-update hook kept for API compatibility (no-op without LDA)."""
        return

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        features = self._to_device(features)

        if len(self.seen_classes) == 0:
            if self.use_gpu:
                return torch.zeros(len(features), dtype=torch.long)
            else:
                return np.zeros(len(features), dtype=int)

        # Whiten features
        features_whitened = self._whiten_features(features)

        # Get class prototypes (exact Welford means, no EMA!)
        if self.use_gpu:
            prototypes = torch.stack(
                [self.class_stats[cls]["mean"] for cls in self.seen_classes]
            )
            prototypes_whitened = self._whiten_features(prototypes)

            # Normalize and compute cosine similarity
            features_norm = features_whitened / (
                torch.norm(features_whitened, dim=1, keepdim=True) + 1e-8
            )
            prototypes_norm = prototypes_whitened / (
                torch.norm(prototypes_whitened, dim=1, keepdim=True) + 1e-8
            )

            scores = features_norm @ prototypes_norm.T
            predictions = torch.argmax(scores, dim=1)

            # Map indices to actual class labels (consistent with CPU branch)
            return np.array(
                [self.seen_classes[pred.cpu().item()] for pred in predictions]
            )
        else:
            prototypes = np.array(
                [self.class_stats[cls]["mean"] for cls in self.seen_classes]
            )
            prototypes_whitened = self._whiten_features(prototypes)

            # Normalize and compute cosine similarity
            features_norm = features_whitened / (
                np.linalg.norm(features_whitened, axis=1, keepdims=True) + 1e-8
            )
            prototypes_norm = prototypes_whitened / (
                np.linalg.norm(prototypes_whitened, axis=1, keepdims=True) + 1e-8
            )

            scores = features_norm @ prototypes_norm.T
            predictions = np.argmax(scores, axis=1)

            return np.array([self.seen_classes[pred] for pred in predictions])

    @property
    def name(self) -> str:
        return "WhitenedNCM"
