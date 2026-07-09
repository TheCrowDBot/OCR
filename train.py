from ocr_training import CRNN, CrossValidator, SampleLoader, TrainingConfig, Vocabulary

# ==================== CONFIG ====================

config = TrainingConfig()
print(config.describe_device())

# ==================== DADOS ====================

real_pairs = SampleLoader.load_real_samples(config.real_dir)
synth_pairs = SampleLoader.load_real_samples(config.synth_dir)
print(f"Reais: {len(real_pairs)} | Sintéticos: {len(synth_pairs)}")

labels_all = [p[1] for p in real_pairs] + [p[1] for p in synth_pairs]

# ==================== VOCABULÁRIO ====================

vocab = Vocabulary.from_labels(labels_all)
print(vocab.summary())
vocab.save(config.vocab_path)
print(f"Vocab guardado: {vocab.vocab_size} caracteres")

# ── Verificação defensiva: seq_len suficiente para o CTC alinhar max_len ──
_tmp_model = CRNN(vocab.num_classes, config.target_h, config.target_w)
_tmp_model.check_seq_len(vocab.max_len)
del _tmp_model

# ==================== VALIDAÇÃO CRUZADA ====================

cross_validator = CrossValidator(config, vocab, real_pairs, synth_pairs)
cross_validator.run()