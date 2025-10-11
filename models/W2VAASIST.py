"""
W2V-AASIST: Wav2Vec2 + AASIST for Audio Anti-Spoofing
Combines Wav2Vec2 self-supervised features with AASIST graph attention architecture
Strict reproduction for SpeechFake dataset baseline
Optimized for A800 GPU training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model
from typing import Dict

# Import AASIST components
from .AASIST import (
    GraphAttentionLayer,
    HtrgGraphAttentionLayer, 
    GraphPool
)


class W2VEncoder(nn.Module):
    """Wav2Vec2 feature extractor to replace AASIST's CONV + ResNet encoder"""
    def __init__(self, w2v_checkpoint: str, freeze: bool = True):
        super().__init__()
        
        print(f"Loading Wav2Vec2 from: {w2v_checkpoint}")
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(w2v_checkpoint)
        
        # Freeze Wav2Vec2 parameters if specified (as per paper: freeze 95% layers, fine-tune last layers)
        if freeze:
            # Freeze feature extractor (conv layers)
            self.wav2vec2.freeze_feature_extractor()
            
            # Freeze most transformer layers, keep only last transformer block trainable
            total_layers = self.wav2vec2.config.num_hidden_layers
            freeze_layers = int(total_layers * 0.95)  # Freeze 95% of layers
            
            for i in range(freeze_layers):
                for param in self.wav2vec2.encoder.layers[i].parameters():
                    param.requires_grad = False
            
            # Keep last transformer block + feature projection trainable
            print(f"✓ Wav2Vec2: frozen {freeze_layers}/{total_layers} transformer layers (95%)")
            print(f"✓ Wav2Vec2: keeping last {total_layers - freeze_layers} layers trainable")
        else:
            print(f"✓ Wav2Vec2: all parameters trainable")
        
        # Wav2Vec2 output dimension (usually 768 for base, 1024 for large)
        self.output_dim = self.wav2vec2.config.hidden_size
        print(f"✓ Wav2Vec2 output dimension: {self.output_dim}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, audio_length) - raw waveform
        Returns:
            features: (batch_size, seq_len, hidden_size)
        """
        outputs = self.wav2vec2(x)
        return outputs.last_hidden_state


class FeatureProjection(nn.Module):
    """
    Project Wav2Vec2 features to match AASIST's expected input dimension
    and create spectro-temporal structure
    """
    def __init__(self, input_dim: int, output_dim: int = 64):
        super().__init__()
        
        self.projection = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.SELU(inplace=True)
        )
        self.output_dim = output_dim
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, hidden_size)
        Returns:
            projected: (batch_size, seq_len, output_dim)
        """
        return self.projection(x)


class Model(nn.Module):
    """
    W2V-AASIST Model
    Architecture: Wav2Vec2 → Feature Projection → AASIST Graph Attention Network
    """
    def __init__(self, d_args: Dict):
        super().__init__()
        
        self.d_args = d_args
        
        # Extract W2V configuration
        w2v_checkpoint = d_args.get("w2v_checkpoint", "facebook/wav2vec2-base")
        freeze_w2v = d_args.get("freeze_w2v", "True")
        freeze_w2v = freeze_w2v if isinstance(freeze_w2v, bool) else freeze_w2v.lower() == "true"
        
        # AASIST configuration (use default values if not specified)
        gat_dims = d_args.get("gat_dims", [64, 32])
        pool_ratios = d_args.get("pool_ratios", [0.5, 0.7, 0.5, 0.5])
        temperatures = d_args.get("temperatures", [2.0, 2.0, 100.0, 100.0])
        
        print(f"\n{'='*60}")
        print(f"Initializing W2V-AASIST Model")
        print(f"{'='*60}")
        print(f"Wav2Vec2 Config:")
        print(f"  - Checkpoint: {w2v_checkpoint}")
        print(f"  - Freeze: {freeze_w2v}")
        print(f"AASIST Config:")
        print(f"  - GAT dims: {gat_dims}")
        print(f"  - Pool ratios: {pool_ratios}")
        print(f"  - Temperatures: {temperatures}")
        print(f"{'='*60}\n")
        
        # 1. Wav2Vec2 Encoder (替代 AASIST 的 CONV + ResNet)
        self.w2v_encoder = W2VEncoder(w2v_checkpoint, freeze=freeze_w2v)
        w2v_dim = self.w2v_encoder.output_dim
        
        # 2. Feature Projection (将 W2V 特征投影到 AASIST 期望的维度)
        # AASIST 原始是 64 维 (filts[-1][-1] = 64)
        aasist_feature_dim = gat_dims[0]  # 通常是 64
        self.feature_projection = FeatureProjection(w2v_dim, aasist_feature_dim)
        
        # 3. AASIST Graph Attention Network Components
        self.drop = nn.Dropout(0.5, inplace=True)
        self.drop_way = nn.Dropout(0.2, inplace=True)
        
        # Positional encoding for spectral nodes
        # W2V 输出的 seq_len 会根据音频长度变化，我们需要动态处理
        # 这里我们不使用固定的 pos_S，而是在 forward 中动态生成
        
        # Learnable master nodes
        self.master1 = nn.Parameter(torch.randn(1, 1, gat_dims[0]))
        self.master2 = nn.Parameter(torch.randn(1, 1, gat_dims[0]))
        
        # Graph Attention Layers
        self.GAT_layer_S = GraphAttentionLayer(
            aasist_feature_dim,
            gat_dims[0],
            temperature=temperatures[0]
        )
        self.GAT_layer_T = GraphAttentionLayer(
            aasist_feature_dim,
            gat_dims[0],
            temperature=temperatures[1]
        )
        
        # Heterogeneous Graph Attention Layers
        self.HtrgGAT_layer_ST11 = HtrgGraphAttentionLayer(
            gat_dims[0], gat_dims[1], temperature=temperatures[2]
        )
        self.HtrgGAT_layer_ST12 = HtrgGraphAttentionLayer(
            gat_dims[1], gat_dims[1], temperature=temperatures[2]
        )
        self.HtrgGAT_layer_ST21 = HtrgGraphAttentionLayer(
            gat_dims[0], gat_dims[1], temperature=temperatures[2]
        )
        self.HtrgGAT_layer_ST22 = HtrgGraphAttentionLayer(
            gat_dims[1], gat_dims[1], temperature=temperatures[2]
        )
        
        # Graph Pooling Layers
        self.pool_S = GraphPool(pool_ratios[0], gat_dims[0], 0.3)
        self.pool_T = GraphPool(pool_ratios[1], gat_dims[0], 0.3)
        self.pool_hS1 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hT1 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hS2 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hT2 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        
        # Output layer
        self.out_layer = nn.Linear(5 * gat_dims[1], 2)
        
        print(f"✓ W2V-AASIST Model initialized successfully!\n")
    
    def forward(self, x: torch.Tensor, Freq_aug: bool = False) -> tuple:
        """
        Forward pass combining Wav2Vec2 and AASIST
        
        Args:
            x: (batch_size, audio_length) - raw waveform
            Freq_aug: bool - frequency augmentation (not used for W2V)
        
        Returns:
            last_hidden: (batch_size, 5 * gat_dims[1]) - embeddings
            output: (batch_size, 2) - logits for [spoof, bonafide]
        """
        batch_size = x.size(0)
        
        # 1. Extract Wav2Vec2 features
        # x: (batch_size, audio_length) -> w2v_features: (batch_size, seq_len, w2v_dim)
        w2v_features = self.w2v_encoder(x)
        
        # 2. Project to AASIST dimension
        # (batch_size, seq_len, w2v_dim) -> (batch_size, seq_len, aasist_dim)
        features = self.feature_projection(w2v_features)
        
        # 3. Create spectro-temporal structure
        # 在 AASIST 中，e 的形状是 (#bs, #filt, #spec, #seq)
        # 对于 W2V，我们需要模拟这个结构
        # features: (batch_size, seq_len, feature_dim)
        # 我们将 seq_len 维度分解为 spectral 和 temporal
        
        # 为了简化，我们直接使用时间序列作为图节点
        # e_T 代表时间节点，e_S 代表频谱节点
        # 这里我们将整个序列视为时间节点
        e_T = features  # (batch_size, seq_len, feature_dim)
        
        # 对于频谱节点，我们可以通过转置或者其他方式获得
        # 这里我们简单地复用相同的特征但进行不同的池化
        e_S = features  # (batch_size, seq_len, feature_dim)
        
        # 4. Apply AASIST's Graph Attention Network
        
        # Spectral GAT (GAT-S)
        gat_S = self.GAT_layer_S(e_S)
        out_S = self.pool_S(gat_S)  # (batch_size, #node_S, gat_dims[0])
        
        # Temporal GAT (GAT-T)
        gat_T = self.GAT_layer_T(e_T)
        out_T = self.pool_T(gat_T)  # (batch_size, #node_T, gat_dims[0])
        
        # 5. Learnable master nodes
        master1 = self.master1.expand(batch_size, -1, -1)
        master2 = self.master2.expand(batch_size, -1, -1)
        
        # 6. Heterogeneous graph inference - Path 1
        out_T1, out_S1, master1 = self.HtrgGAT_layer_ST11(
            out_T, out_S, master=self.master1
        )
        out_S1 = self.pool_hS1(out_S1)
        out_T1 = self.pool_hT1(out_T1)
        
        out_T_aug, out_S_aug, master_aug = self.HtrgGAT_layer_ST12(
            out_T1, out_S1, master=master1
        )
        out_T1 = out_T1 + out_T_aug
        out_S1 = out_S1 + out_S_aug
        master1 = master1 + master_aug
        
        # 7. Heterogeneous graph inference - Path 2
        out_T2, out_S2, master2 = self.HtrgGAT_layer_ST21(
            out_T, out_S, master=self.master2
        )
        out_S2 = self.pool_hS2(out_S2)
        out_T2 = self.pool_hT2(out_T2)
        
        out_T_aug, out_S_aug, master_aug = self.HtrgGAT_layer_ST22(
            out_T2, out_S2, master=master2
        )
        out_T2 = out_T2 + out_T_aug
        out_S2 = out_S2 + out_S_aug
        master2 = master2 + master_aug
        
        # 8. Apply dropout
        out_T1 = self.drop_way(out_T1)
        out_T2 = self.drop_way(out_T2)
        out_S1 = self.drop_way(out_S1)
        out_S2 = self.drop_way(out_S2)
        master1 = self.drop_way(master1)
        master2 = self.drop_way(master2)
        
        # 9. Max pooling between two paths
        out_T = torch.max(out_T1, out_T2)
        out_S = torch.max(out_S1, out_S2)
        master = torch.max(master1, master2)
        
        # 10. Aggregate features
        T_max, _ = torch.max(torch.abs(out_T), dim=1)
        T_avg = torch.mean(out_T, dim=1)
        S_max, _ = torch.max(torch.abs(out_S), dim=1)
        S_avg = torch.mean(out_S, dim=1)
        
        # 11. Concatenate all features
        last_hidden = torch.cat(
            [T_max, T_avg, S_max, S_avg, master.squeeze(1)], dim=1
        )
        
        # 12. Final classification
        last_hidden = self.drop(last_hidden)
        output = self.out_layer(last_hidden)
        
        return last_hidden, output


def enable_gradient_checkpointing(model: Model):
    """
    Enable gradient checkpointing for memory-efficient training on A800
    """
    if hasattr(model.w2v_encoder.wav2vec2, 'gradient_checkpointing_enable'):
        model.w2v_encoder.wav2vec2.gradient_checkpointing_enable()
        print("✓ Gradient checkpointing enabled for Wav2Vec2")
    return model


if __name__ == "__main__":
    # Test the model
    print("\n" + "="*60)
    print("Testing W2V-AASIST Model")
    print("="*60 + "\n")
    
    # Configuration matching SpeechFake_W2V_AASIST.conf
    config = {
        "w2v_checkpoint": "facebook/wav2vec2-base",
        "freeze_w2v": "True",
        "embedding_dim": 256,
        "sample_rate": 16000,
        "nb_samp": 64600,  # ~4 seconds at 16kHz
        "gat_dims": [64, 32],
        "pool_ratios": [0.5, 0.7, 0.5, 0.5],
        "temperatures": [2.0, 2.0, 100.0, 100.0]
    }
    
    # Create model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")
    
    model = Model(config).to(device)
    
    # Print model statistics
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    nb_trainable = sum([
        param.view(-1).size()[0] for param in model.parameters() 
        if param.requires_grad
    ])
    
    print(f"\n{'='*60}")
    print(f"Model Statistics")
    print(f"{'='*60}")
    print(f"Total parameters:      {nb_params:,}")
    print(f"Trainable parameters:  {nb_trainable:,}")
    print(f"Frozen parameters:     {nb_params - nb_trainable:,}")
    print(f"{'='*60}\n")
    
    # Test forward pass
    batch_size = 2
    audio_length = config["nb_samp"]
    dummy_input = torch.randn(batch_size, audio_length).to(device)
    
    print(f"Testing forward pass...")
    print(f"Input shape: {dummy_input.shape}")
    
    try:
        with torch.no_grad():
            embedding, output = model(dummy_input)
        
        print(f"✓ Embedding shape: {embedding.shape}")
        print(f"✓ Output shape: {output.shape}")
        print(f"\n{'='*60}")
        print(f"Model test PASSED! ✓")
        print(f"{'='*60}\n")
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"Model test FAILED! ✗")
        print(f"Error: {e}")
        print(f"{'='*60}\n")
        raise
