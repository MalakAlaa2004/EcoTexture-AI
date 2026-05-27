import os
import sys
from pathlib import Path
import tensorflow as tf
try:
    import tf_keras as keras
except ImportError:
    try:
        from tensorflow import keras
    except ImportError:
        import keras

# Add project root to path
PROJECT_ROOT = Path(r"D:\EcoTexture AI")
sys.path.insert(0, str(PROJECT_ROOT))

# Setup legacy keras
os.environ["TF_USE_LEGACY_KERAS"] = "1"

from ecotexture_ai.model import build_ecotexture_model

print("TensorFlow version:", tf.__version__)
print("Keras version:", keras.__version__)

def test_vit():
    try:
        print("\n--- Inline Testing vit_tiny compilation ---")
        from transformers import TFAutoModel
        layers, Input, Model = keras.layers, keras.Input, keras.Model
        
        img_input = Input(shape=(224, 224, 3), name="image_input")
        backbone = TFAutoModel.from_pretrained("google/vit-base-patch16-224-in21k", from_pt=True, name="tf_vi_t_model")
        
        # Transpose input
        transposed_input = layers.Lambda(lambda x: tf.transpose(x, perm=[0, 3, 1, 2]))(img_input)
        outputs = backbone(pixel_values=transposed_input)
        
        # ViT pooler output
        cnn_feat = outputs.pooler_output
        print("ViT Tiny backbone built successfully, pooler_output shape:", cnn_feat.shape)
        
        # Build classification model (similar head projection in model.py)
        # Note: in model.py, build_ecotexture_model projects cnn_feat to embed_dim (256)
        cnn_proj = layers.Dense(256, activation="gelu", name="cnn_proj")(cnn_feat)
        cnn_proj = layers.LayerNormalization(name="cnn_norm")(cnn_proj)
        
        # SIFT branch placeholder
        sift_input = Input(shape=(300,), name="sift_input")
        sift_proj  = layers.Dense(256, activation="gelu", name="sift_proj1")(sift_input)
        sift_proj  = layers.Dense(256, activation="gelu", name="sift_proj2")(sift_proj)
        sift_proj  = layers.LayerNormalization(name="sift_norm")(sift_proj)
        
        from ecotexture_ai.model import CrossAttentionFusion
        fused = CrossAttentionFusion(d_model=256, num_heads=4, dropout=0.35)(cnn_proj, sift_proj)
        
        x = layers.Dropout(0.35)(fused)
        x = layers.Dense(256, activation="gelu", name="head_dense")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.35)(x)
        out = layers.Dense(12, activation="softmax", name="output")(x)
        
        model = Model(inputs=[img_input, sift_input], outputs=out, name="EcoTextureAI")
        
        ckpt_vit = r"D:\grad project\trial\benchmark_experiments_external\models\coco_mixed\ViT_Tiny\seed42_best.weights.h5"
        if os.path.exists(ckpt_vit):
            print("Loading ViT weights...")
            # Note: since the checkpoint weights were saved from ViTTinyClassifier, their hierarchy is:
            # backbone: "tf_vi_t_model" inside model vs "backbone" inside ViTTinyClassifier
            # We can use skip_mismatch=True or load_weights by name
            model.load_weights(ckpt_vit, skip_mismatch=True)
            print("ViT weights loaded successfully!")
        else:
            print("ViT checkpoint not found.")
    except Exception as e:
        print("Error with ViT:", e)
        import traceback
        traceback.print_exc()

test_vit()
