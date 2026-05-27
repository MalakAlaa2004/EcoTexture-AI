"""
EcoTexture AI — Hybrid Model Architecture
==========================================
CNN (EfficientNetB0) + SIFT BoVW + Cross-Attention Fusion
Original EcoTexture AI architecture for waste material recognition.
"""
from __future__ import annotations

import numpy as np
import tensorflow as tf
try:
    import tf_keras as keras
    from tf_keras import Input, Model, layers, models
except ImportError:
    try:
        from tensorflow import keras
        from tensorflow.keras import Input, Model, layers, models
    except ImportError:
        import keras
        from keras import Input, Model, layers, models

try:
    import keras as standalone_keras
except ImportError:
    standalone_keras = None

from ecotexture_ai.config import EMBED_DIM, IMG_SIZE, NUM_HEADS, SIFT_K

def _get_optimizer(learning_rate: float, weight_decay: float = 0.0):
    try:
        return keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
    except AttributeError:
        return keras.optimizers.Adam(learning_rate=learning_rate)


# ─────────────────────────────────────────────────────────────
# CROSS-ATTENTION FUSION
# ─────────────────────────────────────────────────────────────
@keras.utils.register_keras_serializable(package="EcoTexture", name="CrossAttentionFusion")
class CrossAttentionFusion(layers.Layer):
    """Multi-head cross-attention fusion of CNN and SIFT features."""

    def __init__(self, d_model: int = 256, num_heads: int = 4, dropout: float = 0.3, **kwargs):
        super().__init__(**kwargs)
        self.d_model   = d_model
        self.num_heads = num_heads
        self.dropout_rate = dropout
        self.mha       = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)
        self.norm1     = layers.LayerNormalization(epsilon=1e-6)
        self.norm2     = layers.LayerNormalization(epsilon=1e-6)
        self.ffn       = models.Sequential([
            layers.Dense(d_model * 4, activation="gelu"),
            layers.Dropout(dropout),
            layers.Dense(d_model),
        ])
        self.dropout   = layers.Dropout(dropout)

    def call(self, cnn_feat, sift_feat, training: bool = False):
        cnn_feat  = tf.cast(cnn_feat,  tf.float32)
        sift_feat = tf.cast(sift_feat, tf.float32)
        q   = tf.expand_dims(cnn_feat,  1)
        k   = tf.expand_dims(sift_feat, 1)
        attn_out = self.mha(query=q, value=k, key=k, training=training)
        attn_out = self.dropout(attn_out, training=training)
        out1     = self.norm1(q + attn_out)
        ffn_out  = self.ffn(out1, training=training)
        final    = self.norm2(out1 + ffn_out)
        return tf.squeeze(final, 1)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"d_model": self.d_model, "num_heads": self.num_heads, "dropout": self.dropout_rate})
        return cfg


# ─────────────────────────────────────────────────────────────
# MODEL BUILDER
# ─────────────────────────────────────────────────────────────
def build_ecotexture_model(
    num_classes: int,
    backbone_name: str = "efficientnetb0",
    img_size: tuple[int, int] = IMG_SIZE,
    sift_k: int = SIFT_K,
    embed_dim: int = EMBED_DIM,
    num_heads: int = NUM_HEADS,
    dropout: float = 0.35,
    freeze_backbone: bool = True,
) -> keras.Model:
    """
    Hybrid CNN + SIFT BoVW + Cross-Attention model.
    Stage 1: train head only (backbone frozen).
    Stage 2: fine-tune last N blocks.
    """
    img_input = Input(shape=(*img_size, 3), name="image_input")
    sift_input = Input(shape=(sift_k,), name="sift_input")

    if backbone_name == "custom_cnn":
        # Conv block 1
        x = layers.Conv2D(32, 3, padding="same", activation="relu", name="conv1_conv")(img_input)
        x = layers.BatchNormalization(name="conv1_bn")(x)
        x = layers.MaxPooling2D(name="conv1_pool")(x)
        
        # Conv block 2
        x = layers.Conv2D(64, 3, padding="same", activation="relu", name="conv2_conv")(x)
        x = layers.BatchNormalization(name="conv2_bn")(x)
        x = layers.MaxPooling2D(name="conv2_pool")(x)
        
        # Conv block 3
        x = layers.Conv2D(128, 3, padding="same", activation="relu", name="conv3_conv")(x)
        x = layers.BatchNormalization(name="conv3_bn")(x)
        x = layers.MaxPooling2D(name="conv3_pool")(x)
        
        # Conv block 4
        x = layers.Conv2D(256, 3, padding="same", activation="relu", name="conv4_conv")(x)
        x = layers.BatchNormalization(name="conv4_bn")(x)
        x = layers.GlobalAveragePooling2D(name="conv4_pool")(x)
        
        cnn_feat = layers.Dense(256, activation="relu", name="cnn_dense")(x)
        cnn_feat = layers.Dropout(0.4, name="cnn_dropout")(cnn_feat)
        
        # SIFT branch
        sift_proj = layers.Dense(128, activation="relu", name="sift_dense1")(sift_input)
        sift_proj = layers.BatchNormalization(name="sift_bn")(sift_proj)
        sift_proj = layers.Dense(64, activation="relu", name="sift_dense2")(sift_proj)
        
        # Fusion
        merged = layers.Concatenate(name="fusion_concat")([cnn_feat, sift_proj])
        merged = layers.Reshape((1, 320), name="fusion_reshape")(merged)
        attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=64, name="fusion_attn")(merged, merged)
        x = layers.LayerNormalization(name="fusion_norm")(attn + merged)
        x = layers.Flatten(name="fusion_flatten")(x)
        
        # Classification Head
        z = layers.Dense(128, activation="relu", name="head_dense")(x)
        z = layers.Dropout(0.4, name="head_dropout")(z)
        out = layers.Dense(num_classes, activation="softmax", name="output")(z)
        
        model = Model(inputs=[img_input, sift_input], outputs=out, name="Cross_Attention")
        
        if freeze_backbone:
            # Freeze conv block layers
            for layer in model.layers:
                if any(k in layer.name for k in ["conv1_", "conv2_", "conv3_", "conv4_"]):
                    layer.trainable = False
    else:
        # ── CNN BRANCH ─────────────────────────────────────────────
        if backbone_name == "convnext_tiny":
            from transformers import TFConvNextModel
            backbone = TFConvNextModel.from_pretrained("facebook/convnext-tiny-224", name="tf_conv_next_model")
            backbone.trainable = not freeze_backbone
            # Transpose input from NHWC to NCHW for Hugging Face model
            transposed_input = layers.Lambda(lambda x: tf.transpose(x, perm=[0, 3, 1, 2]))(img_input)
            cnn_feat = backbone(transposed_input).pooler_output
        else:
            backbone_map = {
                "efficientnetb0": keras.applications.EfficientNetB0,
                "efficientnetb2": keras.applications.EfficientNetB2,
                "mobilenetv3":    keras.applications.MobileNetV3Small,
            }
            BackboneCls = backbone_map.get(backbone_name, keras.applications.EfficientNetB0)
            backbone = BackboneCls(
                include_top=False, weights="imagenet",
                input_tensor=img_input,
                pooling="avg",
            )
            backbone.trainable = not freeze_backbone
            cnn_feat = backbone.output                             # (B, C)
        cnn_proj = layers.Dense(embed_dim, activation="gelu", name="cnn_proj")(cnn_feat)
        cnn_proj = layers.LayerNormalization(name="cnn_norm")(cnn_proj)

        # ── SIFT BRANCH ────────────────────────────────────────────
        sift_proj  = layers.Dense(embed_dim,     activation="gelu", name="sift_proj1")(sift_input)
        sift_proj  = layers.Dense(embed_dim,     activation="gelu", name="sift_proj2")(sift_proj)
        sift_proj  = layers.LayerNormalization(name="sift_norm")(sift_proj)

        # ── CROSS-ATTENTION FUSION ─────────────────────────────────
        fused = CrossAttentionFusion(d_model=embed_dim, num_heads=num_heads, dropout=dropout)(
            cnn_proj, sift_proj
        )

        # ── CLASSIFICATION HEAD ────────────────────────────────────
        x = layers.Dropout(dropout)(fused)
        x = layers.Dense(embed_dim, activation="gelu", name="head_dense")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout)(x)
        out = layers.Dense(num_classes, activation="softmax", name="output")(x)

        model = Model(inputs=[img_input, sift_input], outputs=out, name="EcoTextureAI")

    model.compile(
        optimizer=_get_optimizer(learning_rate=1e-3, weight_decay=1e-4),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    return model


def load_weights_topologically(model: keras.Model, checkpoint_path: str):
    """Loads weights from a saved keras weights file or model file by matching layer shapes directly."""
    import h5py
    with h5py.File(checkpoint_path, 'r') as f:
        # Check if it is a weights-only file or full model file
        weight_group = f
        if 'model_weights' in f:
            weight_group = f['model_weights']
        
        # Get list of layer names in h5 file
        h5_layer_names = list(weight_group.keys())
        
        # Filter layers in our current model that have weights
        model_layers = [l for l in model.layers if len(l.weights) > 0]
        
        # Match weight matrices by comparing shapes sequentially
        copied = 0
        h5_idx = 0
        
        for tgt in model_layers:
            tgt_shapes = [w.shape for w in tgt.weights]
            
            # Find the next layer in h5 file that matches this layer's weight shapes
            while h5_idx < len(h5_layer_names):
                h5_layer_name = h5_layer_names[h5_layer_name_idx := h5_idx]
                h5_idx += 1
                
                layer_g = weight_group[h5_layer_name]
                # Extract weight datasets
                weight_names = []
                def get_dataset_names(name, obj):
                    if isinstance(obj, h5py.Dataset):
                        weight_names.append(name)
                layer_g.visititems(get_dataset_names)
                
                # Sort weight names by internal hierarchy or name to align
                weight_names = sorted(weight_names, key=lambda x: (x.count('/'), x))
                src_weights = [np.array(layer_g[name]) for name in weight_names]
                src_shapes = [w.shape for w in src_weights]
                
                if len(tgt_shapes) == len(src_shapes) and all(t == s for t, s in zip(tgt_shapes, src_shapes)):
                    tgt.set_weights(src_weights)
                    copied += 1
                    break
                    
        print(f"[EcoTexture] Sequentially loaded {copied}/{len(model_layers)} layers from weights file {checkpoint_path}")



def build_contrastive_transfer_model(
    num_classes: int,
    checkpoint_path: str | None = None,
    img_size: tuple[int, int] = IMG_SIZE,
    sift_k: int = SIFT_K,
    backbone_name: str = "efficientnetb0",
) -> keras.Model:
    """
    Transfer learning from COCO/MET research checkpoint.
    Loads the saved cross-attention backbone and replaces the classification head
    with a waste-material head.
    """
    model = build_ecotexture_model(
        num_classes=num_classes,
        img_size=img_size,
        sift_k=sift_k,
        backbone_name=backbone_name,
        freeze_backbone=True,
    )
    if checkpoint_path:
        try:
            if backbone_name == "custom_cnn":
                # Use robust topological loading for custom_cnn checkpoints
                try:
                    load_weights_topologically(model, checkpoint_path)
                except Exception as topo_err:
                    print(f"[EcoTexture] Topological weight load failed: {topo_err}. Trying standard load...")
                    model.load_weights(checkpoint_path, by_name=True, skip_mismatch=True)
            else:
                if checkpoint_path.endswith(".weights.h5"):
                    try:
                        model.load_weights(checkpoint_path, skip_mismatch=True)
                    except ValueError:
                        model.load_weights(checkpoint_path, by_name=True, skip_mismatch=True)
                else:
                    model.load_weights(checkpoint_path, by_name=True, skip_mismatch=True)
                print(f"[EcoTexture] Loaded weights from: {checkpoint_path}")
        except Exception as exc:
            print(f"[EcoTexture] WARNING - could not load checkpoint: {exc}")
    return model


def unfreeze_last_blocks(model: keras.Model, train_last_n: int = 30, learning_rate: float = 1e-5):
    """Gradual unfreezing for stage-2 fine-tuning."""
    for layer in model.layers:
        layer.trainable = True

    # Refreeze everything except the last N layers
    for layer in model.layers[:-train_last_n]:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = False

    model.compile(
        optimizer=_get_optimizer(learning_rate=learning_rate, weight_decay=1e-5),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    print(f"[EcoTexture] Stage-2: unfroze last {train_last_n} layers, lr={learning_rate}")

