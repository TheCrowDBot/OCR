# Optical Character Recognition (OCR) for E-R Diagrams

**OCR** is a Python pipeline for reading text (entity and attribute names) from hand-drawn and printed **Entity-Relationship diagrams**.

It combines a **YOLO** detector (to locate entity/attribute boxes) with a custom **CRNN + CTC** model (to read the text inside each box), trained on real diagram crops augmented with synthetic, font-rendered word images.

---

## ✨ Features

- End-to-end pipeline: dataset export → synthetic data generation → training, all in one command
- Real crops exported directly from Label Studio (YOLO OBB format)
- Synthetic data generator that renders real vocabulary in multiple handwriting-style fonts
- CRNN + CTC model with k-fold cross-validation
- Configurable image augmentation (rotation, ruled lines, brightness/contrast jitter, etc.)
- Ready-to-use inference script: YOLO detection + OCR on a single photo

---

## 📦 Installation

```bash
poetry install
```

---

## 🚀 Quick Example

Run the full pipeline (export + crop → synthetic fonts → training):

```bash
poetry run python run_pipeline.py
```

Optional flags:

```bash
poetry run python run_pipeline.py --skip-synthetic                    # sem passo de fontes sintéticas
poetry run python run_pipeline.py --skip-export --skip-synthetic      # só treino
poetry run python run_pipeline.py --fonts fonts --variations 3        # personalizar fontes/variações
```

Test the trained model on a photo:

```bash
poetry run python test_model.py --image foto.jpg
```

---

## 🧱 Core Concepts

### `run_pipeline.py`
Orquestra a pipeline completa: export do dataset, geração de dados sintéticos e treino do modelo, correndo cada etapa como um subprocesso e parando se alguma falhar.

### `export_and_crop.py`
Vai buscar o dataset ao Label Studio (imagens + anotações YOLO OBB) e faz crop das entidades e atributos anotados, guardando-os em `real_crops/`.

### `generate_words_fonts_v3.py`
Lê as palavras presentes em `real_crops/` e renderiza-as com **8 fontes diferentes**, gerando variações sintéticas guardadas em `synthetic_crops/` para aumentar o dataset de treino.

### `train.py`
Constrói o vocabulário a partir das amostras reais e sintéticas, e treina o modelo **CRNN + CTC** com validação cruzada (`ocr_training/cross_validator.py`), guardando o melhor modelo e o vocabulário em `results/`.

### `test_model.py`
Carrega o modelo treinado (`results/best_model_final.pt`), corre o **YOLO** para detetar entidades/atributos numa foto e aplica o OCR a cada crop detetado, mostrando o resultado.

### `visualize_augmentations/visualize_aug.py`
Mostra a augmentation utilizada no treino: gera uma grelha com cada transformação isolada (valores exagerados para leitura em relatório) e outra com o pipeline completo replicado tal como usado em `OCRTrainDataset`.

---

## 🧠 Model & Training

- **Arquitetura:** CRNN (Convolutional Recurrent Neural Network) treinado com **CTC loss**
- **Validação:** k-fold cross-validation (10 folds por defeito)
- **Tamanho de imagem:** 256×50 (largura × altura), normalizado antes de entrar no modelo
- **Early stopping** e *LR scheduler* (`ReduceLROnPlateau`) configurados em `ocr_training/config.py`
- Deteção das entidades/atributos na imagem original feita por um modelo **YOLO** separado, previamente treinado

---

## 📤 Output

A pipeline gera, em `results/`:

- `best_model_final.pt` — pesos do melhor modelo CRNN
- `vocab.json` — vocabulário de caracteres usado pelo modelo

O `test_model.py` produz uma imagem anotada com os resultados do OCR sobre a foto de entrada.

---

## ⚙️ Development

```bash
poetry install
```

Correr a pipeline completa:

```bash
poetry run python run_pipeline.py
```

---

## 📁 Project Structure

```
OCR-main/
├── run_pipeline.py                 # Orquestra a pipeline completa
├── export_and_crop.py              # Export do Label Studio + crop
├── generate_words_fonts_v3.py      # Geração de crops sintéticos com fontes
├── train.py                        # Treino do modelo CRNN + CTC
├── test_model.py                   # Inferência: YOLO + OCR numa foto
├── ocr_training/
│   ├── model.py                    # Arquitetura CRNN
│   ├── dataset.py                  # Dataset de treino/validação
│   ├── augmentation.py             # Transformações de augmentation
│   ├── config.py                   # Configuração central de treino
│   ├── vocabulary.py               # Vocabulário de caracteres
│   ├── decoder.py                  # Descodificação CTC
│   ├── metrics.py                  # Métricas de avaliação
│   ├── cross_validator.py          # Validação cruzada
│   └── data_utils.py               # Utilitários de dados
├── visualize_augmentations/
│   └── visualize_aug.py            # Visualização das augmentations
└── results/                        # Modelo treinado, vocabulário e outputs
```

---

## 🧪 Use Cases

- Leitura automática de nomes de entidades e atributos em diagramas E-R desenhados à mão
- Digitalização de diagramas de esquema de bases de dados
- Apoio à conversão de diagramas em papel/quadro para formato digital

---

## 👥 Authors

- Rúben Costa
- Miguel Dias