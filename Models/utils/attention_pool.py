import torch
import torch.nn as nn


class AttentionPool(nn.Module):
    """
    k‑token multi‑head attention aggregator (Set‑Transformer style).
    If `n_tokens = 1` this is identical to the old single‑token pool.
    """
    def __init__(self, d_model: int, n_heads: int = 4, n_tokens: int = 4):
        super().__init__()
        self.n_tokens = n_tokens
        self.query_tokens = nn.Parameter(torch.randn(1, n_tokens, d_model))
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, N, d_model)   encoded sensors
        returns pooled : (B, n_tokens * d_model) after flattening
        """
        B = x.size(0)
        q = self.query_tokens.expand(B, -1, -1)         # (B, k, d)
        pooled, _ = self.attn(q, x, x)                  # (B, k, d)
        return pooled.flatten(1)                        # (B, k·d) 


class StatisticalPool(nn.Module):
    """
    Pure statistical aggregation pool that computes explicit statistical features.
    """
    def __init__(self, d_model: int, stats: list = None):
        super().__init__()
        self.d_model = d_model
        if stats is None:
            stats = ['mean', 'std', 'min', 'max']
        self.stats = stats
        self.output_dim = len(stats) * d_model
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, N, d_model)   encoded sensors
        returns : (B, len(stats) * d_model)
        """
        features = []
        
        for stat in self.stats:
            if stat == 'mean':
                features.append(torch.mean(x, dim=1))
            elif stat == 'std':
                features.append(torch.std(x, dim=1))
            elif stat == 'min':
                features.append(torch.min(x, dim=1)[0])
            elif stat == 'max':
                features.append(torch.max(x, dim=1)[0])
            elif stat == 'median':
                features.append(torch.median(x, dim=1)[0])
            elif stat == 'sum':
                features.append(torch.sum(x, dim=1))
            else:
                raise ValueError(f"Unknown statistic: {stat}")
        
        return torch.cat(features, dim=1)


class HybridAttentionPool(nn.Module):
    """
    Combines attention-based pooling with explicit statistical features.
    """
    def __init__(self, d_model: int, n_heads: int = 4, n_tokens: int = 2, 
                 stats: list = None, combine_strategy: str = 'concat'):
        super().__init__()
        self.d_model = d_model
        self.combine_strategy = combine_strategy
        
        # Attention component
        self.n_tokens = n_tokens
        self.query_tokens = nn.Parameter(torch.randn(1, n_tokens, d_model))
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        
        # Statistical component
        if stats is None:
            stats = ['mean', 'std']
        self.stats = stats
        self.stat_pool = StatisticalPool(d_model, stats)
        
        # Combination layer if needed
        if combine_strategy == 'learned':
            total_dim = n_tokens * d_model + len(stats) * d_model
            self.combine_layer = nn.Linear(total_dim, n_tokens * d_model)
        elif combine_strategy == 'weighted':
            self.attention_weight = nn.Parameter(torch.tensor(0.5))
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, N, d_model)   encoded sensors
        returns : (B, output_dim)
        """
        B = x.size(0)
        
        # Attention features
        q = self.query_tokens.expand(B, -1, -1)
        attn_features, _ = self.attn(q, x, x)  # (B, n_tokens, d_model)
        attn_features = attn_features.flatten(1)  # (B, n_tokens * d_model)
        
        # Statistical features
        stat_features = self.stat_pool(x)  # (B, len(stats) * d_model)
        
        # Combine features
        if self.combine_strategy == 'concat':
            return torch.cat([attn_features, stat_features], dim=1)
        elif self.combine_strategy == 'learned':
            combined = torch.cat([attn_features, stat_features], dim=1)
            return self.combine_layer(combined)
        elif self.combine_strategy == 'weighted':
            # Ensure dimensions match for weighted combination
            if attn_features.shape[1] != stat_features.shape[1]:
                raise ValueError("For weighted combination, attention and statistical features must have same dimension")
            weight = torch.sigmoid(self.attention_weight)
            return weight * attn_features + (1 - weight) * stat_features
        else:
            raise ValueError(f"Unknown combine strategy: {self.combine_strategy}")


class SpecializedAttentionPool(nn.Module):
    """
    Attention pool where each query token is specialized for different aspects.
    Each token is initialized and constrained to focus on different statistical properties.
    """
    def __init__(self, d_model: int, n_heads: int = 4, specializations: list = None):
        super().__init__()
        if specializations is None:
            specializations = ['general', 'extreme', 'variance', 'local']
        
        self.specializations = specializations
        self.n_tokens = len(specializations)
        self.d_model = d_model
        
        # Initialize different query tokens for different purposes
        self.query_tokens = nn.Parameter(torch.randn(1, self.n_tokens, d_model))
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        
        # Optional: Add regularization to encourage specialization
        self.use_specialization_loss = True
        
    def _init_specialized_queries(self):
        """Initialize query tokens based on their specialization."""
        with torch.no_grad():
            for i, spec in enumerate(self.specializations):
                if spec == 'general':
                    # Default initialization
                    pass
                elif spec == 'extreme':
                    # Initialize to be more sensitive to outliers
                    self.query_tokens[0, i] *= 2.0
                elif spec == 'variance':
                    # Initialize differently for variance detection
                    self.query_tokens[0, i] = torch.randn_like(self.query_tokens[0, i]) * 0.1
                elif spec == 'local':
                    # Initialize for local patterns
                    self.query_tokens[0, i] = torch.randn_like(self.query_tokens[0, i]) * 0.5
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, N, d_model)   encoded sensors
        returns : (B, n_tokens * d_model)
        """
        B = x.size(0)
        q = self.query_tokens.expand(B, -1, -1)
        pooled, attn_weights = self.attn(q, x, x)  # (B, n_tokens, d_model)
        
        # Store attention weights for analysis if needed
        self.last_attention_weights = attn_weights
        
        return pooled.flatten(1)
    
    def get_specialization_loss(self):
        """
        Compute a regularization loss to encourage query specialization.
        This encourages different queries to attend to different parts of the input.
        """
        if not hasattr(self, 'last_attention_weights') or self.last_attention_weights is None:
            return torch.tensor(0.0)
        
        # Encourage diversity in attention patterns across different queries
        attn = self.last_attention_weights  # (B, n_tokens, N)
        
        # Compute pairwise similarities between attention patterns
        similarities = []
        for i in range(self.n_tokens):
            for j in range(i + 1, self.n_tokens):
                sim = torch.cosine_similarity(attn[:, i, :], attn[:, j, :], dim=1)
                similarities.append(sim.mean())
        
        if similarities:
            # Penalize high similarities (encourage diversity)
            return torch.stack(similarities).mean()
        else:
            return torch.tensor(0.0)


class StructuredStatisticalPool(nn.Module):
    """
    Statistical pool that processes each statistic separately before combining.
    This preserves the semantic meaning of each statistic.
    """
    def __init__(self, d_model: int, stats: list = None, fusion_strategy: str = 'separate_then_combine'):
        super().__init__()
        self.d_model = d_model
        if stats is None:
            stats = ['mean', 'std', 'min', 'max']
        self.stats = stats
        self.fusion_strategy = fusion_strategy
        
        if fusion_strategy == 'separate_then_combine':
            # Each statistic gets its own processing layer
            self.stat_processors = nn.ModuleDict()
            for stat in stats:
                self.stat_processors[stat] = nn.Sequential(
                    nn.Linear(d_model, d_model // 2),
                    nn.ReLU(),
                    nn.Linear(d_model // 2, d_model // 2)
                )
            self.output_dim = len(stats) * (d_model // 2)
            
        elif fusion_strategy == 'attention_fusion':
            # Use attention to combine statistics
            self.stat_attention = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)
            self.output_dim = d_model
            
        elif fusion_strategy == 'weighted_sum':
            # Learn weights for each statistic
            self.stat_weights = nn.Parameter(torch.ones(len(stats)))
            self.output_dim = d_model
            
        else:
            raise ValueError(f"Unknown fusion strategy: {fusion_strategy}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, N, d_model)   encoded sensors
        returns : (B, output_dim)
        """
        # Compute all statistics
        stat_features = {}
        for stat in self.stats:
            if stat == 'mean':
                stat_features[stat] = torch.mean(x, dim=1)
            elif stat == 'std':
                stat_features[stat] = torch.std(x, dim=1)
            elif stat == 'min':
                stat_features[stat] = torch.min(x, dim=1)[0]
            elif stat == 'max':
                stat_features[stat] = torch.max(x, dim=1)[0]
            elif stat == 'median':
                stat_features[stat] = torch.median(x, dim=1)[0]
            elif stat == 'sum':
                stat_features[stat] = torch.sum(x, dim=1)
        
        if self.fusion_strategy == 'separate_then_combine':
            # Process each statistic separately
            processed_features = []
            for stat in self.stats:
                processed = self.stat_processors[stat](stat_features[stat])
                processed_features.append(processed)
            return torch.cat(processed_features, dim=1)
            
        elif self.fusion_strategy == 'attention_fusion':
            # Stack statistics and use attention
            stat_stack = torch.stack([stat_features[stat] for stat in self.stats], dim=1)  # (B, n_stats, d_model)
            
            # Self-attention over statistics
            attended, _ = self.stat_attention(stat_stack, stat_stack, stat_stack)  # (B, n_stats, d_model)
            
            # Global average pooling over statistics
            return torch.mean(attended, dim=1)  # (B, d_model)
            
        elif self.fusion_strategy == 'weighted_sum':
            # Weighted combination of statistics
            weights = torch.softmax(self.stat_weights, dim=0)
            stat_stack = torch.stack([stat_features[stat] for stat in self.stats], dim=1)  # (B, n_stats, d_model)
            
            # Apply weights
            weighted = stat_stack * weights.view(1, -1, 1)  # Broadcasting
            return torch.sum(weighted, dim=1)  # (B, d_model)


class ScaleAwareStatisticalPool(nn.Module):
    """
    Statistical pool that normalizes each statistic before combining.
    This handles scale differences between statistics.
    """
    def __init__(self, d_model: int, stats: list = None):
        super().__init__()
        self.d_model = d_model
        if stats is None:
            stats = ['mean', 'std', 'min', 'max']
        self.stats = stats
        
        # Learnable normalization parameters for each statistic
        self.stat_normalizers = nn.ModuleDict()
        for stat in stats:
            self.stat_normalizers[stat] = nn.LayerNorm(d_model)
        
        self.output_dim = len(stats) * d_model
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, N, d_model)   encoded sensors
        returns : (B, len(stats) * d_model)
        """
        normalized_features = []
        
        for stat in self.stats:
            # Compute statistic
            if stat == 'mean':
                feature = torch.mean(x, dim=1)
            elif stat == 'std':
                feature = torch.std(x, dim=1)
            elif stat == 'min':
                feature = torch.min(x, dim=1)[0]
            elif stat == 'max':
                feature = torch.max(x, dim=1)[0]
            elif stat == 'median':
                feature = torch.median(x, dim=1)[0]
            elif stat == 'sum':
                feature = torch.sum(x, dim=1)
            
            # Normalize each statistic separately
            normalized = self.stat_normalizers[stat](feature)
            normalized_features.append(normalized)
        
        return torch.cat(normalized_features, dim=1) 