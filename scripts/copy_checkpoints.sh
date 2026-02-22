#!/bin/bash

# Create base checkpoints directory
CHECKPOINT_BASE="/home/kkor/cxt/checkpoints"
mkdir -p "$CHECKPOINT_BASE"

# Create subdirectories for each model type
mkdir -p "$CHECKPOINT_BASE/broad"
mkdir -p "$CHECKPOINT_BASE/broad+adapter"
mkdir -p "$CHECKPOINT_BASE/narrow"
mkdir -p "$CHECKPOINT_BASE/broad_w200"
mkdir -p "$CHECKPOINT_BASE/residual"
mkdir -p "$CHECKPOINT_BASE/w200_wmissing"
mkdir -p "$CHECKPOINT_BASE/w200_wmissing_adapter"

# Copy broad model
echo "Copying broad model checkpoint..."
cp /sietch_colab/data_share/cxt/models/broad/version_20/checkpoints/epoch=1-step=5280.ckpt \
   "$CHECKPOINT_BASE/broad/broad_epoch=1-step=5280.ckpt"

# Copy broad+adapter model
echo "Copying broad+adapter model checkpoint..."
cp /sietch_colab/data_share/cxt/models/broad+adapter/version_26/checkpoints/epoch=2-step=792.ckpt \
   "$CHECKPOINT_BASE/broad+adapter/broad_adapter_epoch=2-step=792.ckpt"

# Copy narrow model
echo "Copying narrow model checkpoint..."
cp /home/kkor/cxt/cxt/lightning_logs/version_47/checkpoints/epoch=5-step=4692.ckpt \
   "$CHECKPOINT_BASE/narrow/narrow_epoch=5-step=4692.ckpt"

# Copy broad_w200 model
echo "Copying broad_w200 model checkpoint..."
cp /sietch_colab/data_share/cxt/models/broad_w200/version_29/checkpoints/epoch=1-step=944.ckpt \
   "$CHECKPOINT_BASE/broad_w200/broad_w200_epoch=1-step=944.ckpt"

# Copy residual model
echo "Copying residual model checkpoint..."
cp /sietch_colab/data_share/cxt/models/residual/version_46/checkpoints/epoch=1-step=5280.ckpt \
   "$CHECKPOINT_BASE/residual/residual_epoch=1-step=5280.ckpt"

# Copy w200_wmissing model
echo "Copying w200_wmissing model checkpoint..."
cp /sietch_colab/data_share/cxt/models/w200_wmissing/version_48/checkpoints/epoch=1-step=944.ckpt \
   "$CHECKPOINT_BASE/w200_wmissing/w200_wmissing_epoch=1-step=944.ckpt"

# Copy w200_wmissing_adapter model
echo "Copying w200_wmissing_adapter model checkpoint..."
cp /home/kkor/cxt/cxt/lightning_logs/version_50/checkpoints/epoch=9-step=480.ckpt \
   "$CHECKPOINT_BASE/w200_wmissing_adapter/w200_wmissing_adapter_epoch=9-step=480.ckpt"

echo "All checkpoints copied successfully!"
echo "Directory structure:"
tree "$CHECKPOINT_BASE" -L 2
