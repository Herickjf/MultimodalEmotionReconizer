# Arquitetura Multimodal Self-Attention para Reconhecimento de Emoções

> Documento de referência para implementação completa do projeto.

## 1. Objetivo

Construir um sistema de reconhecimento de emoções a partir de **vídeo de uma pessoa falando**, utilizando três modalidades:

- Texto (conteúdo semântico)
- Áudio (prosódia e características acústicas)
- Expressão facial (dinâmica temporal)

A arquitetura é composta por três grandes blocos:

1. Extração de características (backbones pré-treinados)
2. Fusão multimodal (implementação própria)
3. Classificação (implementação própria)

---

# 2. Pipeline

```text
Vídeo
 ├── Whisper → Texto → BERT
 ├── Áudio → Wav2Vec2
 └── Frames → RetinaFace → VideoMAE

Embeddings
      ↓
Linear Projections
      ↓
Encoder Multimodal (Transformer)
      ↓
Pooling
      ↓
MLP
      ↓
Softmax
```

---

# 3. Organização recomendada

```text
project/
│
├── notebooks/
│   ├── 01_preprocess.ipynb
│   └── 02_training.ipynb
│
├── cache/
│   ├── text_embeddings/
│   ├── audio_embeddings/
│   └── image_embeddings/
│
├── checkpoints/
├── datasets/
├── src/
└── configs/
```

---

# 4. Bibliotecas

```bash
pip install torch torchvision torchaudio transformers timm
pip install opencv-python ffmpeg-python
pip install accelerate datasets evaluate
pip install torchmetrics scikit-learn
pip install matplotlib pandas tqdm
```

Importações principais:

```python
import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoModel,
    Wav2Vec2Processor,
    Wav2Vec2Model,
    WhisperProcessor,
    WhisperForConditionalGeneration,
)
```

---

# 5. Backbones (não implementar)

## Texto
- Whisper (ASR)
- BERT/RoBERTa

Congelar:

```python
for p in bert.parameters():
    p.requires_grad=False
```

## Áudio
- Wav2Vec2 Base

Congelar igualmente.

## Imagem

Pipeline:

Vídeo → RetinaFace → Sequência de rostos → VideoMAE

Congelar inicialmente.

---

# 6. Embeddings

Padronizar dimensões.

Exemplo:

- Texto 768
- Áudio 1024
- Imagem 768

Projetar para:

```python
EMBED_DIM=512
```

```python
self.text_proj=nn.Linear(768,512)
self.audio_proj=nn.Linear(1024,512)
self.image_proj=nn.Linear(768,512)
```

Essas camadas DEVEM ser treinadas.

---

# 7. Encoder Multimodal (principal contribuição)

Implementar utilizando TransformerEncoder.

Sugestão:

- embed_dim = 512
- heads = 8
- ff_dim = 2048
- dropout = 0.1
- layers = 4

```python
encoder_layer=nn.TransformerEncoderLayer(
    d_model=512,
    nhead=8,
    dim_feedforward=2048,
    dropout=0.1,
    batch_first=True
)

encoder=nn.TransformerEncoder(
    encoder_layer,
    num_layers=4
)
```

Entrada:

```text
[Texto][Áudio][Imagem]
```

Saída:

Representação multimodal.

---

# 8. Pooling

Preferência:

1. CLS Token
2. Mean Pooling

Mean:

```python
x=x.mean(dim=1)
```

CLS:

Adicionar parâmetro treinável.

---

# 9. MLP

```python
nn.Sequential(
    nn.Linear(512,256),
    nn.ReLU(),
    nn.Dropout(.3),
    nn.Linear(256,128),
    nn.ReLU(),
    nn.Dropout(.2),
    nn.Linear(128,n_classes)
)
```

---

# 10. Treinamento

Treinar SOMENTE:

- projeções lineares
- encoder
- pooling parametrizado
- MLP

Não treinar:

- Whisper
- BERT
- Wav2Vec2
- VideoMAE

Optimizer:

```python
optimizer=torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=1e-2
)
```

Loss:

```python
criterion=nn.CrossEntropyLoss()
```

Scheduler recomendado:

CosineAnnealingLR.

Gradient clipping:

```python
torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
```

Mixed Precision:

```python
torch.cuda.amp.autocast()
```

---

# 11. Pré-processamento

Notebook separado.

Executar uma única vez.

- Extrair áudio
- Transcrever
- Extrair faces
- Gerar embeddings
- Salvar cache

Formato:

```
cache/
    sample01.pt
    sample02.pt
```

Cada arquivo:

```python
{
"text":text_embedding,
"audio":audio_embedding,
"image":image_embedding,
"label":label
}
```

---

# 12. Notebook 1

Responsável por:

- dataset
- ffmpeg
- whisper
- retinaface
- videomae
- cache

Nunca treinar.

---

# 13. Notebook 2

Responsável por:

- carregar cache
- encoder
- pooling
- mlp
- treinamento
- métricas

---

# 14. Métricas

Accuracy

Precision

Recall

F1 Macro

Confusion Matrix

---

# 15. Cuidados

- Balancear classes.
- Padding consistente.
- Mesmo número de frames.
- Não recalcular embeddings a cada época.
- Salvar checkpoints.
- Fixar seeds.
- Validar shapes constantemente.

---

# 16. Roadmap

1. Preparar dataset
2. Extrair áudio
3. Transcrever
4. Detectar rostos
5. Gerar embeddings
6. Salvar cache
7. Construir projeções
8. Construir encoder
9. Construir pooling
10. Construir MLP
11. Treinar
12. Validar
13. Testar
14. Fine-tuning parcial dos backbones (opcional)

Este documento serve como especificação-base do projeto e pode ser utilizado para contextualizar sobre a arquitetura, responsabilidades de cada módulo e fluxo completo de implementação.
