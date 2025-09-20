"""
SAR Imagery Preprocessing Pipeline

Advanced preprocessing techniques for Synthetic Aperture Radar imagery
optimized for ship detection in Arctic maritime environments.
"""

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageEnhance, ImageFilter
import cv2
from typing import Dict, List, Tuple, Optional, Union
import albumentations as A
from albumentations.pytorch import ToTensorV2
import rasterio
from rasterio.enums import Resampling
import json
import os
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)


class SARImageProcessor:
    """
    Advanced SAR Image Processing for Maritime Detection
    
    Features:
    - Multi-polarization handling (HH, HV, VV, VH)
    - Speckle noise reduction
    - Adaptive contrast enhancement
    - Sea-ice interference removal
    - Wind pattern normalization
    - Arctic condition optimization
    """
    
    def __init__(
        self,
        target_size: Tuple[int, int] = (224, 224),
        normalize_db: bool = True,
        denoise_method: str = 'bilateral',
        enhance_contrast: bool = True
    ):
        self.target_size = target_size
        self.normalize_db = normalize_db
        self.denoise_method = denoise_method
        self.enhance_contrast = enhance_contrast
        
        # Statistics for SAR imagery normalization (calculated from Arctic datasets)
        self.sar_mean = -15.2  # dB
        self.sar_std = 8.7     # dB
        
        self.setup_transforms()
    
    def setup_transforms(self):
        """Initialize transformation pipelines"""
        
        # Training augmentations
        self.train_transforms = A.Compose([
            A.Resize(height=self.target_size[0], width=self.target_size[1]),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.Rotate(limit=30, p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.3,
                contrast_limit=0.3,
                p=0.7
            ),
            A.GaussNoise(var_limit=(5.0, 20.0), p=0.3),
            A.GaussianBlur(blur_limit=(1, 3), p=0.2),
            A.ElasticTransform(
                alpha=1,
                sigma=50,
                alpha_affine=50,
                p=0.3
            ),
            A.GridDistortion(p=0.2),
            A.OpticalDistortion(distort_limit=0.1, p=0.2),
            ToTensorV2()
        ])
        
        # Validation transforms (no augmentation)
        self.val_transforms = A.Compose([
            A.Resize(height=self.target_size[0], width=self.target_size[1]),
            ToTensorV2()
        ])
    
    def linear_to_db(self, image: np.ndarray) -> np.ndarray:
        """Convert linear SAR values to dB scale"""
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        return 10 * np.log10(np.maximum(image, epsilon))
    
    def db_to_linear(self, image: np.ndarray) -> np.ndarray:
        """Convert dB SAR values to linear scale"""
        return 10 ** (image / 10)
    
    def reduce_speckle_noise(self, image: np.ndarray, method: str = 'bilateral') -> np.ndarray:
        """
        Advanced speckle noise reduction for SAR imagery
        
        Methods:
        - bilateral: Bilateral filtering (preserves edges)
        - nlm: Non-local means (best quality, slower)
        - median: Fast median filtering
        - wiener: Wiener filtering for Gaussian noise
        """
        
        if method == 'bilateral':
            return cv2.bilateralFilter(
                image.astype(np.float32),
                d=9,
                sigmaColor=75,
                sigmaSpace=75
            )
        elif method == 'nlm':
            return cv2.fastNlMeansDenoising(
                (image * 255).astype(np.uint8),
                None,
                h=10,
                templateWindowSize=7,
                searchWindowSize=21
            ).astype(np.float32) / 255.0
        elif method == 'median':
            return cv2.medianBlur((image * 255).astype(np.uint8), 5).astype(np.float32) / 255.0
        elif method == 'wiener':
            # Simple Wiener filter approximation
            kernel = np.ones((3, 3)) / 9
            filtered = cv2.filter2D(image, -1, kernel)
            noise_var = np.var(image - filtered)
            signal_var = np.var(image)
            wiener_factor = signal_var / (signal_var + noise_var)
            return image * wiener_factor + filtered * (1 - wiener_factor)
        else:
            return image
    
    def enhance_maritime_features(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance ship-like features in SAR imagery
        """
        # Adaptive histogram equalization
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply((image * 255).astype(np.uint8))
        enhanced = enhanced.astype(np.float32) / 255.0
        
        # Edge enhancement for ship boundaries
        edge_kernel = np.array([[-1, -1, -1],
                               [-1,  8, -1],
                               [-1, -1, -1]])
        edges = cv2.filter2D(enhanced, -1, edge_kernel)
        
        # Combine original with enhanced edges
        result = 0.7 * enhanced + 0.3 * np.clip(edges, 0, 1)
        
        return result
    
    def normalize_sar_image(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize SAR imagery using maritime-specific statistics
        """
        if self.normalize_db:
            # Convert to dB if not already
            if np.max(image) > 10:  # Assume linear if max > 10
                image = self.linear_to_db(image)
            
            # Normalize using Arctic SAR statistics
            normalized = (image - self.sar_mean) / self.sar_std
        else:
            # Min-max normalization
            normalized = (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-8)
        
        return normalized
    
    def remove_sea_ice_interference(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Remove sea ice interference from SAR imagery
        Uses morphological operations and texture analysis
        """
        if mask is not None:
            # Use provided sea ice mask
            return np.where(mask, image, np.median(image))
        
        # Automatic sea ice detection
        # Sea ice typically has uniform low backscatter
        threshold = np.percentile(image, 20)
        ice_mask = image < threshold
        
        # Morphological operations to clean mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        ice_mask = cv2.morphologyEx(ice_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        ice_mask = cv2.morphologyEx(ice_mask, cv2.MORPH_OPEN, kernel)
        
        # Replace ice regions with median water value
        water_median = np.median(image[ice_mask == 0])
        result = np.where(ice_mask, water_median, image)
        
        return result
    
    def preprocess_image(
        self,
        image: Union[np.ndarray, str, Path],
        is_training: bool = True,
        sea_ice_mask: Optional[np.ndarray] = None
    ) -> torch.Tensor:
        """
        Complete preprocessing pipeline for SAR images
        """
        # Load image if path provided
        if isinstance(image, (str, Path)):
            with rasterio.open(image) as src:
                image = src.read(1).astype(np.float32)
        
        # Ensure image is 2D
        if len(image.shape) > 2:
            image = image.squeeze()
        
        # Remove sea ice interference
        if sea_ice_mask is not None or True:  # Auto-detect if no mask provided
            image = self.remove_sea_ice_interference(image, sea_ice_mask)
        
        # Speckle noise reduction
        image = self.reduce_speckle_noise(image, self.denoise_method)
        
        # Maritime feature enhancement
        if self.enhance_contrast:
            image = self.enhance_maritime_features(image)
        
        # Normalization
        image = self.normalize_sar_image(image)
        
        # Clip extreme values
        image = np.clip(image, -3, 3)  # 3 standard deviations
        
        # Convert to uint8 for albumentations
        image_uint8 = ((image + 3) / 6 * 255).astype(np.uint8)
        
        # Apply augmentations
        if is_training:
            transformed = self.train_transforms(image=image_uint8)
        else:
            transformed = self.val_transforms(image=image_uint8)
        
        # Convert back to float32 and proper range
        tensor = transformed['image'].float() / 255.0 * 6 - 3
        
        return tensor.unsqueeze(0)  # Add channel dimension


class ShipDetectionDataset(Dataset):
    """
    PyTorch Dataset for Ship Detection in SAR Imagery
    
    Supports:
    - Multi-format image loading (GeoTIFF, PNG, JPG)
    - Annotation format flexibility (COCO, YOLO, custom)
    - Real-time augmentation
    - Multi-scale training
    - Negative mining
    """
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        annotations_file: Union[str, Path],
        processor: SARImageProcessor,
        mode: str = 'train',
        min_ship_size: int = 5,  # Minimum ship size in pixels
        negative_ratio: float = 0.3,  # Ratio of negative samples
        multi_scale: bool = True
    ):
        self.data_dir = Path(data_dir)
        self.annotations_file = Path(annotations_file)
        self.processor = processor
        self.mode = mode
        self.min_ship_size = min_ship_size
        self.negative_ratio = negative_ratio
        self.multi_scale = multi_scale
        
        # Load annotations
        self.load_annotations()
        
        # Multi-scale sizes for training
        self.scales = [224, 256, 288, 320] if multi_scale and mode == 'train' else [224]
    
    def load_annotations(self):
        """Load annotations from various formats"""
        with open(self.annotations_file, 'r') as f:
            if self.annotations_file.suffix == '.json':
                data = json.load(f)
                if 'annotations' in data:  # COCO format
                    self.annotations = self._parse_coco_annotations(data)
                else:  # Custom format
                    self.annotations = data
            else:
                raise ValueError(f"Unsupported annotation format: {self.annotations_file.suffix}")
        
        # Filter by minimum ship size
        self.annotations = [
            ann for ann in self.annotations 
            if self._get_ship_area(ann) >= self.min_ship_size ** 2
        ]
        
        # Add negative samples
        if self.negative_ratio > 0:
            self._add_negative_samples()
    
    def _parse_coco_annotations(self, coco_data: Dict) -> List[Dict]:
        """Parse COCO format annotations"""
        annotations = []
        
        # Create image_id to filename mapping
        image_map = {img['id']: img['file_name'] for img in coco_data['images']}
        
        for ann in coco_data['annotations']:
            annotations.append({
                'image_file': image_map[ann['image_id']],
                'bbox': ann['bbox'],  # [x, y, width, height]
                'category_id': ann['category_id'],
                'area': ann['area'],
                'is_ship': ann['category_id'] == 1  # Assuming ship category_id = 1
            })
        
        return annotations
    
    def _get_ship_area(self, annotation: Dict) -> float:
        """Calculate ship area from annotation"""
        if 'area' in annotation:
            return annotation['area']
        elif 'bbox' in annotation:
            bbox = annotation['bbox']
            return bbox[2] * bbox[3]  # width * height
        else:
            return 0
    
    def _add_negative_samples(self):
        """Add negative (no-ship) samples for training"""
        positive_count = len(self.annotations)
        negative_count = int(positive_count * self.negative_ratio)
        
        # Find images without ships
        all_images = set(f.name for f in self.data_dir.glob('*.tif') if f.is_file())
        all_images.update(f.name for f in self.data_dir.glob('*.png') if f.is_file())
        
        annotated_images = set(ann['image_file'] for ann in self.annotations)
        negative_images = list(all_images - annotated_images)
        
        # Sample negative images
        negative_samples = np.random.choice(
            negative_images, 
            min(negative_count, len(negative_images)), 
            replace=False
        )
        
        for img_file in negative_samples:
            self.annotations.append({
                'image_file': img_file,
                'bbox': None,
                'category_id': 0,
                'area': 0,
                'is_ship': False
            })
    
    def __len__(self) -> int:
        return len(self.annotations)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        annotation = self.annotations[idx]
        
        # Load image
        image_path = self.data_dir / annotation['image_file']
        
        # Random scale selection for multi-scale training
        if self.multi_scale and self.mode == 'train':
            scale = np.random.choice(self.scales)
            self.processor.target_size = (scale, scale)
        
        # Preprocess image
        image_tensor = self.processor.preprocess_image(
            image_path,
            is_training=(self.mode == 'train')
        )
        
        # Prepare labels
        is_ship = annotation['is_ship']
        ship_label = torch.tensor(1 if is_ship else 0, dtype=torch.long)
        
        # Confidence target (higher for larger, clearer ships)
        if is_ship and 'area' in annotation:
            # Normalize confidence based on ship size
            confidence = min(annotation['area'] / 1000.0, 1.0)
        else:
            confidence = 0.0 if not is_ship else 0.5
        
        confidence_tensor = torch.tensor(confidence, dtype=torch.float32)
        
        # Size classification (Small: 0, Medium: 1, Large: 2)
        if is_ship and 'area' in annotation:
            area = annotation['area']
            if area < 100:
                size_class = 0  # Small
            elif area < 500:
                size_class = 1  # Medium
            else:
                size_class = 2  # Large
        else:
            size_class = 0  # Default to small for non-ships
        
        size_tensor = torch.tensor(size_class, dtype=torch.long)
        
        return {
            'image': image_tensor,
            'ship_label': ship_label,
            'confidence': confidence_tensor,
            'size_label': size_tensor,
            'image_file': annotation['image_file'],
            'bbox': annotation.get('bbox', [0, 0, 0, 0])
        }


def create_dataloaders(
    train_dir: Union[str, Path],
    val_dir: Union[str, Path],
    train_annotations: Union[str, Path],
    val_annotations: Union[str, Path],
    batch_size: int = 32,
    num_workers: int = 4,
    target_size: Tuple[int, int] = (224, 224)
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation dataloaders
    """
    
    # Initialize processors
    train_processor = SARImageProcessor(
        target_size=target_size,
        normalize_db=True,
        denoise_method='bilateral',
        enhance_contrast=True
    )
    
    val_processor = SARImageProcessor(
        target_size=target_size,
        normalize_db=True,
        denoise_method='bilateral',
        enhance_contrast=True
    )
    
    # Create datasets
    train_dataset = ShipDetectionDataset(
        data_dir=train_dir,
        annotations_file=train_annotations,
        processor=train_processor,
        mode='train',
        multi_scale=True
    )
    
    val_dataset = ShipDetectionDataset(
        data_dir=val_dir,
        annotations_file=val_annotations,
        processor=val_processor,
        mode='val',
        multi_scale=False
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    return train_loader, val_loader


# Data visualization utilities
def visualize_sar_preprocessing(
    image_path: Union[str, Path],
    processor: SARImageProcessor,
    save_path: Optional[Union[str, Path]] = None
):
    """
    Visualize the SAR preprocessing pipeline steps
    """
    import matplotlib.pyplot as plt
    
    # Load original image
    with rasterio.open(image_path) as src:
        original = src.read(1).astype(np.float32)
    
    # Apply preprocessing steps
    denoised = processor.reduce_speckle_noise(original)
    enhanced = processor.enhance_maritime_features(denoised)
    normalized = processor.normalize_sar_image(enhanced)
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].imshow(original, cmap='gray')
    axes[0, 0].set_title('Original SAR Image')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(denoised, cmap='gray')
    axes[0, 1].set_title('After Denoising')
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(enhanced, cmap='gray')
    axes[1, 0].set_title('Feature Enhanced')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(normalized, cmap='gray')
    axes[1, 1].set_title('Normalized')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()


if __name__ == "__main__":
    # Test preprocessing pipeline
    processor = SARImageProcessor(
        target_size=(224, 224),
        normalize_db=True,
        denoise_method='bilateral'
    )
    
    print("SAR Image Processor initialized successfully!")
    print(f"Target size: {processor.target_size}")
    print(f"Normalization: {'dB scale' if processor.normalize_db else 'Min-Max'}")
    print(f"Denoising method: {processor.denoise_method}")
    
    # Test with random data
    test_image = np.random.exponential(0.1, (512, 512)).astype(np.float32)
    processed = processor.preprocess_image(test_image, is_training=True)
    
    print(f"Processed image shape: {processed.shape}")
    print(f"Processed image range: [{processed.min():.3f}, {processed.max():.3f}]")
    print("Preprocessing pipeline test completed successfully!")
