# Visual Caption Generator

<div align="center">
  <img src="https://github.com/asghar-rizvi/Image-Caption-Generator/raw/main/Output/Screenshot%202025-04-12%20144839.png" width="600" alt="Caption Example">
</div>## Architecture

```mermaid
graph LR
    A[Image] --> B[VGG16-CNN]
    B --> C[Feature Tensor]
    C --> D[Attention Layer]
    D --> E[LSTM Decoder]
    E --> F[Caption Tokens]
```

## Core Components

### 1. Vision Encoder
**Modified VGG16**:
- Last FC layer replaced with 256D bottleneck  
- Added spatial attention gates  
- Fine-tuned conv5 block  

### 2. Language Decoder
**2-Layer LSTM**:
- 512 hidden units  
- Bahdanau attention  
- Beam search (k=3)  

## Training Specs

| Parameter       | Value               |
|-----------------|---------------------|
| Dataset         | Flickr30k (145K)    |
| Epochs          | 30                  |
| Batch Size      | 64                  |
| Optimizer       | AdamW (lr=3e-4)     |
| Loss            | Label Smoothing     |

## Output
<div align="center">
  <img src="https://github.com/asghar-rizvi/Image-Caption-Generator/blob/main/Output/Screenshot%202025-04-12%20144852.png" width="600" alt="Caption Example">
</div>
