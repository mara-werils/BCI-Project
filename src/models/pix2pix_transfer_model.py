"""
Pix2Pix model with Transfer Learning and HER2 Classification support.
Extends the base Pix2PixModel with:
- Pre-trained weights loading
- Encoder freezing for fine-tuning
- HER2 classification head
- Multi-task learning (generation + classification)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .pix2pix_model import Pix2PixModel
from . import networks


class HER2ClassificationHead(nn.Module):
    """
    Classification head for HER2 status prediction.
    Extracts features from generator and predicts HER2 class.
    """
    
    def __init__(self, input_channels=256, num_classes=4, dropout=0.5):
        super().__init__()
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_channels, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, features):
        """
        Args:
            features: Feature map from generator [B, C, H, W]
        Returns:
            logits: Classification logits [B, num_classes]
        """
        x = self.pool(features)
        x = self.classifier(x)
        return x


class Pix2PixTransferModel(Pix2PixModel):
    """
    Extended Pix2Pix model with transfer learning and classification.
    """
    
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add model-specific options."""
        parser = Pix2PixModel.modify_commandline_options(parser, is_train)
        return parser
    
    def __init__(self, opt):
        """Initialize the model with transfer learning support."""
        # Initialize base model
        Pix2PixModel.__init__(self, opt)
        
        # Track frozen state
        self.encoder_frozen = False
        self.current_epoch = 0
        
        # Load pre-trained weights if specified
        if opt.isTrain and opt.pretrained_path:
            self._load_pretrained_generator(opt.pretrained_path)
        
        if opt.isTrain and opt.pretrained_D_path:
            self._load_pretrained_discriminator(opt.pretrained_D_path)
        
        # Apply encoder freezing if specified
        if opt.isTrain and opt.freeze_encoder:
            self._freeze_encoder()
        
        # Add classification head if enabled
        if opt.enable_classification:
            self._setup_classification_head(opt)
        
        # Adjust learning rate for fine-tuning
        if opt.isTrain and opt.pretrained_path and opt.finetune_lr_factor != 1.0:
            for optimizer in self.optimizers:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= opt.finetune_lr_factor
            print(f"Adjusted learning rate by factor {opt.finetune_lr_factor}")
    
    def _load_pretrained_generator(self, path):
        """Load pre-trained generator weights."""
        print(f"Loading pre-trained generator from: {path}")
        
        try:
            state_dict = torch.load(path, map_location=self.device)
            
            # Handle DataParallel wrapper
            if hasattr(self.netG, 'module'):
                model_dict = self.netG.module.state_dict()
            else:
                model_dict = self.netG.state_dict()
            
            # Filter compatible weights
            pretrained_dict = {}
            for k, v in state_dict.items():
                # Remove 'module.' prefix if present
                clean_key = k.replace('module.', '')
                if clean_key in model_dict and v.shape == model_dict[clean_key].shape:
                    pretrained_dict[clean_key] = v
            
            # Load weights
            model_dict.update(pretrained_dict)
            
            if hasattr(self.netG, 'module'):
                self.netG.module.load_state_dict(model_dict)
            else:
                self.netG.load_state_dict(model_dict)
            
            print(f"Loaded {len(pretrained_dict)}/{len(model_dict)} layers from pre-trained model")
            
        except Exception as e:
            print(f"Warning: Could not load pre-trained weights: {e}")
    
    def _load_pretrained_discriminator(self, path):
        """Load pre-trained discriminator weights."""
        print(f"Loading pre-trained discriminator from: {path}")
        
        try:
            state_dict = torch.load(path, map_location=self.device)
            
            if hasattr(self.netD, 'module'):
                self.netD.module.load_state_dict(state_dict)
            else:
                self.netD.load_state_dict(state_dict)
            
            print("Loaded pre-trained discriminator")
            
        except Exception as e:
            print(f"Warning: Could not load pre-trained discriminator: {e}")
    
    def _freeze_encoder(self):
        """Freeze encoder layers of generator."""
        print("Freezing encoder layers...")
        
        frozen_count = 0
        
        # Get the actual model (handle DataParallel)
        model = self.netG.module if hasattr(self.netG, 'module') else self.netG
        
        # For ResNet generator: freeze first half of layers
        if hasattr(model, 'model'):
            layers = list(model.model.children())
            n_freeze = len(layers) // 2
            
            for i, layer in enumerate(layers[:n_freeze]):
                for param in layer.parameters():
                    param.requires_grad = False
                    frozen_count += 1
        
        self.encoder_frozen = True
        print(f"Frozen {frozen_count} parameters in encoder")
    
    def _unfreeze_encoder(self):
        """Unfreeze all encoder layers."""
        print("Unfreezing encoder layers...")
        
        model = self.netG.module if hasattr(self.netG, 'module') else self.netG
        
        for param in model.parameters():
            param.requires_grad = True
        
        self.encoder_frozen = False
        print("All parameters are now trainable")
    
    def _setup_classification_head(self, opt):
        """Setup HER2 classification head."""
        print("Setting up HER2 classification head...")
        
        # Add classification to model names
        self.model_names.append('Classifier')
        
        # Add loss name
        self.loss_names.append('G_classifier')
        
        # Determine feature dimension based on generator architecture
        if 'resnet' in opt.netG:
            # ResNet generator: features at bottleneck
            feature_dim = opt.ngf * 4  # Typically 256 for ngf=64
        else:
            # U-Net generator
            feature_dim = opt.ngf * 8  # Typically 512 for ngf=64
        
        # Create classification head
        self.netClassifier = HER2ClassificationHead(
            input_channels=feature_dim,
            num_classes=opt.num_classes,
            dropout=0.5
        ).to(self.device)
        
        # Loss function
        if opt.class_weighted_loss:
            # Weights will be set from src.dataset
            self.criterion_classifier = nn.CrossEntropyLoss()
        else:
            self.criterion_classifier = nn.CrossEntropyLoss()
        
        # Optimizer
        if opt.isTrain:
            self.optimizer_C = torch.optim.Adam(
                self.netClassifier.parameters(),
                lr=opt.lr * 0.5,  # Lower LR for classifier
                betas=(opt.beta1, 0.999)
            )
            self.optimizers.append(self.optimizer_C)
        
        # Initialize feature extractor hook
        self._setup_feature_hook()
        
        print(f"Classification head created with {opt.num_classes} classes")
    
    def _setup_feature_hook(self):
        """Setup hook to extract features from generator."""
        self.extracted_features = None
        
        def hook_fn(module, input, output):
            self.extracted_features = output
        
        # Get the model
        model = self.netG.module if hasattr(self.netG, 'module') else self.netG
        
        # Register hook at bottleneck layer
        if hasattr(model, 'model'):
            # ResNet generator
            layers = list(model.model.children())
            mid_idx = len(layers) // 2
            if mid_idx < len(layers):
                layers[mid_idx].register_forward_hook(hook_fn)
        else:
            # For other architectures, hook at a suitable layer
            print("Warning: Feature extraction hook may not work for this architecture")
    
    def set_input(self, input):
        """Unpack input data including HER2 labels."""
        Pix2PixModel.set_input(self, input)
        
        # Get HER2 label if available
        if 'her2_label' in input:
            self.her2_label = input['her2_label'].to(self.device)
            self.has_her2_label = True
        else:
            self.has_her2_label = False
    
    def forward(self):
        """Forward pass with optional classification."""
        # Generate fake image
        Pix2PixModel.forward(self)
        
        # Classify if enabled
        if hasattr(self, 'netClassifier') and self.extracted_features is not None:
            self.her2_logits = self.netClassifier(self.extracted_features)
            self.her2_pred = torch.argmax(self.her2_logits, dim=1)
    
    def backward_G(self):
        """Calculate losses including classification."""
        # Original generation losses
        Pix2PixModel.backward_G(self)
        
        # Classification loss
        if (hasattr(self, 'netClassifier') and 
            self.has_her2_label and 
            hasattr(self, 'her2_logits')):
            
            self.loss_G_classifier = self.criterion_classifier(
                self.her2_logits, 
                self.her2_label
            ) * self.opt.lambda_classifier
            
            self.loss_G_classifier.backward()
    
    def optimize_parameters(self, fixD=False):
        """Optimization step with epoch-based unfreezing."""
        # Check if we should unfreeze encoder
        if (self.encoder_frozen and 
            self.opt.freeze_epochs > 0 and 
            self.current_epoch >= self.opt.freeze_epochs):
            self._unfreeze_encoder()
        
        # Forward pass
        self.forward()
        
        # Update D
        if 'noGAN' not in self.opt.pattern and not fixD:
            self.set_requires_grad(self.netD, True)
            self.optimizer_D.zero_grad()
            self.backward_D()
            self.optimizer_D.step()
        
        # Update G and Classifier
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        
        if hasattr(self, 'optimizer_C'):
            self.optimizer_C.zero_grad()
        
        self.backward_G()
        
        self.optimizer_G.step()
        
        if hasattr(self, 'optimizer_C'):
            self.optimizer_C.step()
    
    def set_epoch(self, epoch):
        """Set current epoch for unfreezing logic."""
        self.current_epoch = epoch
    
    def get_classification_accuracy(self):
        """Calculate classification accuracy for current batch."""
        if not (hasattr(self, 'her2_pred') and self.has_her2_label):
            return 0.0
        
        correct = (self.her2_pred == self.her2_label).sum().item()
        total = self.her2_label.size(0)
        
        return correct / total if total > 0 else 0.0
    
    def save_networks(self, epoch):
        """Save networks including classifier."""
        Pix2PixModel.save_networks(self, epoch)
        
        # Save classifier if exists
        if hasattr(self, 'netClassifier'):
            save_filename = f'{epoch}_net_Classifier.pth'
            save_path = os.path.join(self.save_dir, save_filename)
            
            if hasattr(self.netClassifier, 'module'):
                torch.save(self.netClassifier.module.state_dict(), save_path)
            else:
                torch.save(self.netClassifier.state_dict(), save_path)


# Import os for save path
import os


