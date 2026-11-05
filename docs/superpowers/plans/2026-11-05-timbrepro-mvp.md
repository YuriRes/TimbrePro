# TimbrePro MVP: plano de implementação

> **Para agentes:** SUB-SKILL OBRIGATÓRIA: use superpowers:subagent-driven-development (recomendado) ou superpowers:executing-plans para implementar este plano task a task. Os passos usam checkbox (`- [ ]`) para acompanhamento.

**Objetivo:** fechar o fluxo ponta a ponta "música → preset JSON → tocar ao vivo" até 12/11/2026.

**Arquitetura:** a IA só escreve um preset de 11 números; o som ao vivo continua todo na `FxChain`. O dataset é gerado pela mesma `FxChain` (DI + parâmetros sorteados → som) e uma CNN sobre espectrogramas mel aprende o caminho inverso. Na inferência, o Demucs isola a guitarra, a CNN roda em 8 janelas e a mediana vira o preset.

**Stack:** Python 3.10+, numpy, scipy, soundfile, sounddevice; PyTorch só em `model.py` e `train.py`; Demucs só em `match.py` (chamado como processo externo).

**Spec:** documento "TimbrePro — Metodologia e plano de 1 semana" (https://claude.ai/artifact/2x9gVCffDjgWK8cov5saQa), seções "Metodologia, etapa por etapa" e "Plano de 7 dias".

## Restrições globais

- Taxa de amostragem `SAMPLE_RATE = 48_000` e ordem dos parâmetros `PARAM_NAMES`, ambas de `timbrepro/params.py`. Nunca redefinir.
- Parâmetros sempre normalizados em [0, 1], vetor `float64` de forma `(11,)`.
- Trechos de treino: 3 s de clipe + 1 s de pré-rolagem.
- Mel: `n_fft = 2048`, `hop = 512`, 128 bandas, `fmax = 16_000` Hz, em dB. Um clipe de 3 s vira uma matriz `(128, 282)`.
- `dataset.py` e `features.py` não importam `torch` (o dataset é gerado sem GPU).
- Dados, áudios e pesos ficam em `data/` e `checkpoints/`, ignorados pelo git.
- Textos, docstrings e mensagens em português, no estilo do código existente.
- Rodar `python -m pytest` a partir da raiz do repositório (`TimbrePro/TimbrePro`).

## Pontos de atenção para a revisão

1. **Arquivo DI curto ou silencioso:** `split_clips` devolve lista vazia em vez de quebrar. *(teste na Task 1)*
2. **Render com entrada silenciosa:** `render_example` devolve zeros, nunca NaN, ao normalizar o pico. *(teste na Task 1)*
3. **Mel de silêncio:** `log_mel` devolve valores finitos, sem `-inf`. *(teste na Task 2)*
4. **Música mais curta que 8 janelas:** `pick_windows` usa as janelas que houver; com menos de 3 s, completa com zeros. *(teste na Task 7)*
5. **Preset previsto fora de [0, 1] por erro numérico:** `write_preset` aplica `clip` antes de gravar, e o arquivo gravado passa por `load_preset`. *(teste na Task 7)*

---

### Task 1: Sorteio de parâmetros e render dos exemplos

**Arquivos:**
- Criar: `timbrepro/dataset.py`
- Teste: `tests/test_dataset.py`

**Interfaces:**
- Consome: `FxChain(params).process(x)` de `timbrepro/chain.py`; `PARAM_NAMES`, `SAMPLE_RATE` de `timbrepro/params.py`.
- Produz:
  - `CLIP_S = 3`, `PREROLL_S = 1`, `MIX_OFF_PROB = 0.3`, `PEAK = 0.9`
  - `sample_params(rng: np.random.Generator) -> np.ndarray`: forma `(11,)`, uniforme em [0, 1]; com probabilidade `MIX_OFF_PROB`, zera `delay_mix`, e, de forma independente, zera `reverb_mix`.
  - `split_clips(audio: np.ndarray, sr: int = SAMPLE_RATE, min_rms_db: float = -40.0) -> list[np.ndarray]`: janelas consecutivas sem sobreposição, de `(PREROLL_S + CLIP_S) * sr` amostras, avançando `CLIP_S * sr`. Descarta a janela cujo trecho final de `CLIP_S` tem RMS abaixo de `min_rms_db` dBFS.
  - `render_example(dry: np.ndarray, params: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray`: processa `dry` inteiro numa `FxChain` nova, devolve só as últimas `CLIP_S * sr` amostras, normalizadas para pico `PEAK`. Com pico menor que `1e-9`, devolve zeros.

- [ ] **Passo 1: escrever os testes que falham**

```python
def test_sample_params_shape_and_range():          # 1000 sorteios: forma (11,), tudo em [0, 1]
def test_sample_params_turns_mixes_off_sometimes(): # 2000 sorteios: fração de delay_mix == 0 e de reverb_mix == 0 entre 0.2 e 0.4
def test_sample_params_is_reproducible():          # mesma semente → arrays iguais
def test_split_clips_window_length_and_count():    # 10 s de ruído → janelas começando em 0, 3 e 6 s: 3 janelas, cada uma com 4 * SR amostras
def test_split_clips_skips_silent_windows():       # ruído em 0–4 s, silêncio em 4–13 s → só 1 janela
def test_split_clips_short_audio_gives_nothing():  # 2 s de áudio → []
def test_render_example_length_and_peak():         # len == 3 * SR, pico ≈ 0.9 (abs=1e-6)
def test_render_example_uses_the_preroll():        # impulso na amostra 0, reverb_mix=1, reverb_size=1, resto 0 → saída não nula
def test_render_example_of_silence_is_zero():      # entrada zerada → saída zerada, sem NaN
```

- [ ] **Passo 2:** `python -m pytest tests/test_dataset.py -v`. Esperado: FAIL na importação de `timbrepro.dataset`.
- [ ] **Passo 3:** implementar as três funções acima em `timbrepro/dataset.py`.
- [ ] **Passo 4:** `python -m pytest -q`. Esperado: tudo passa.
- [ ] **Passo 5:** commit `feat: sorteio de parâmetros e render de exemplos do dataset`.

### Task 2: Espectrograma mel em numpy

**Arquivos:**
- Criar: `timbrepro/features.py`
- Teste: `tests/test_features.py`

**Interfaces:**
- Consome: `SAMPLE_RATE`.
- Produz:
  - `N_FFT = 2048`, `HOP = 512`, `N_MELS = 128`, `FMAX = 16_000`, `TOP_DB = 80.0`
  - `mel_filterbank(sr: int = SAMPLE_RATE, n_fft: int = N_FFT, n_mels: int = N_MELS, fmax: float = FMAX) -> np.ndarray`: forma `(n_mels, n_fft // 2 + 1)`, triângulos na escala mel HTK (`2595 * log10(1 + f / 700)`), normalização de área (Slaney).
  - `log_mel(audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray`: STFT com janela de Hann e padding reflexivo de `N_FFT // 2` nos dois lados (quadros = `1 + len(audio) // HOP`), potência, mel, `10 * log10(max(x, 1e-10))`, piso em `max - TOP_DB`. Saída em `float32`.

- [ ] **Passo 1: escrever os testes que falham**

```python
def test_filterbank_shape_and_no_empty_band():   # forma (128, 1025), tudo >= 0, toda linha com soma > 0
def test_log_mel_shape_for_a_three_second_clip(): # 3 * SR amostras → (128, 282), dtype float32
def test_sine_peaks_at_its_mel_band():            # seno de 1 kHz: argmax da média no tempo == banda cujo pico do filtro está mais perto de 1 kHz
def test_silence_is_finite():                     # zeros → np.isfinite(tudo)
def test_dynamic_range_is_capped():               # max - min <= TOP_DB + 1e-3
```

- [ ] **Passo 2:** `python -m pytest tests/test_features.py -v`. Esperado: FAIL na importação.
- [ ] **Passo 3:** implementar em `timbrepro/features.py`, usando `numpy.lib.stride_tricks.sliding_window_view` e `np.fft.rfft` (sem librosa).
- [ ] **Passo 4:** `python -m pytest -q`. Esperado: tudo passa.
- [ ] **Passo 5:** commit `feat: espectrograma mel em numpy`.

### Task 3: Gerador do dataset em disco

**Arquivos:**
- Modificar: `timbrepro/dataset.py`, `timbrepro/__main__.py`
- Teste: `tests/test_dataset.py`

**Interfaces:**
- Consome: Task 1 (`split_clips`, `sample_params`, `render_example`), Task 2 (`log_mel`), `read_mono` de `__main__.py` (mover para `timbrepro/audio.py` e importar de lá nos dois lugares).
- Produz:
  - `load_di_clips(di_dir: Path) -> list[np.ndarray]`: lê todo `.wav`/`.flac` da pasta, recursivo, e junta os `split_clips`.
  - `generate(clips, count: int, out_dir: Path, seed: int = 0, workers: int = 4, shard_size: int = 1000) -> None`: o exemplo `i` usa `np.random.default_rng([seed, i])` para sortear o clipe e os parâmetros (reprodutível com qualquer número de workers). Grava `shard_XXXX.npz` com `mels` (`float16`, `(n, 128, 282)`) e `params` (`float32`, `(n, 11)`), e `stats.npz` com média e desvio dos mels (escalares).
  - CLI: `python -m timbrepro dataset --di-dir data/di --out data/dataset --count 20000 --workers 4 --seed 0`. Sem `--di-dir`, usa `synth_di()` (só para teste).
- Depende de `soundfile` instalado (`pip install -r requirements.txt`).

- [ ] **Passo 1:** testes: `test_generate_writes_shards` (`count=5`, `shard_size=2`, clipes de ruído → 3 shards, 5 linhas no total, formas e dtypes acima); `test_generate_is_reproducible_across_workers` (`workers=1` e `workers=2` dão `params` iguais).
- [ ] **Passo 2:** rodar, ver falhar.
- [ ] **Passo 3:** implementar com `multiprocessing.Pool` (em Windows, `if __name__ == "__main__"` já existe em `__main__.py`).
- [ ] **Passo 4:** rodar tudo; gerar um piloto: `python -m timbrepro dataset --count 1000 --out data/piloto` e ouvir 5 exemplos (salvar wav com uma flag `--save-audio 5`).
- [ ] **Passo 5:** commit.

### Task 4: Medição de latência por loopback

**Arquivos:**
- Criar: `timbrepro/latency.py`
- Modificar: `timbrepro/__main__.py`
- Teste: `tests/test_latency.py`

**Interfaces:**
- Produz:
  - `make_click(sr: int = SAMPLE_RATE) -> np.ndarray`: 0,5 s de silêncio, um chirp de 10 ms (1 a 10 kHz, `scipy.signal.chirp`), 1 s de silêncio.
  - `estimate_delay(sent: np.ndarray, received: np.ndarray, sr: int = SAMPLE_RATE) -> float`: atraso em ms pelo pico da correlação cruzada (`scipy.signal.correlate`, modo `full`, método `fft`).
  - `measure(device, block_size: int, repeats: int = 5) -> list[float]`: `sd.playrec` do clique, devolve os atrasos em ms. Importa `sounddevice` dentro da função.
  - CLI: `python -m timbrepro latency --device 12,10 --block 256` imprime média e pior caso.
- [ ] Testes: atraso conhecido de 480 amostras → 10,0 ms (±0,05); com ruído branco de amplitude 0,05 somado, idem; sinal recebido zerado levanta `ValueError("nenhum clique detectado")` (pico abaixo de 10× a mediana da correlação).
- [ ] Rodar, implementar, rodar, commit.
- [ ] Manual (precisa de hardware): cabo da saída 1 na entrada 1, rodar com MME, WASAPI e ASIO; anotar no README.

### Task 5: CNN e perda ponderada

**Arquivos:**
- Criar: `timbrepro/model.py`
- Modificar: `requirements.txt` (adicionar `torch>=2.3`)
- Teste: `tests/test_model.py` (pular o módulo com `pytest.importorskip("torch")`)

**Interfaces:**
- Produz:
  - `TimbreCNN(n_params: int = 11)`: 5 blocos `Conv2d 3×3 → BatchNorm2d → ReLU → MaxPool2d(2)`, canais 32, 64, 128, 256, 256; `AdaptiveAvgPool2d(1)`, `Linear(256, 11)`, sigmoide. Entrada `(B, 1, 128, 282)`.
  - `weighted_mse(pred: Tensor, target: Tensor) -> Tensor`: MSE por parâmetro; os erros de `delay_time` e `delay_feedback` são multiplicados por `target[:, delay_mix]`, o de `reverb_size` por `target[:, reverb_mix]`. Índices tirados de `PARAM_NAMES.index(...)`.
  - `make_resnet18(n_params: int = 11)`: `torchvision.models.resnet18(weights=None)` com `conv1` de 1 canal e `fc` para 11 + sigmoide.
- [ ] Testes: saída `(4, 11)` em [0, 1] para entrada aleatória; com `delay_mix = 0` no alvo, mudar o `delay_time` previsto não muda a perda; perda zero quando `pred == target`.
- [ ] Rodar, implementar, rodar, commit.

### Task 6: Treino e linhas de base

**Arquivos:**
- Criar: `timbrepro/train.py`, `notebooks/treino_colab.ipynb` (só chama `train.main`)
- Teste: `tests/test_train.py` (`importorskip("torch")`)

**Interfaces:**
- Consome: shards e `stats.npz` da Task 3; `TimbreCNN`, `weighted_mse`, `make_resnet18` da Task 5.
- Produz:
  - `ShardDataset(paths, stats, augment: bool)`: devolve `(mel padronizado (1,128,282) float32, params float32)`. Com `augment=True`: deslocamento circular no tempo de até 32 quadros e uma máscara de frequência (até 16 bandas) e uma de tempo (até 24 quadros) com o valor mínimo.
  - `baseline_mae(train_params, test_params) -> dict[str, np.ndarray]`: MAE por parâmetro das linhas de base `media` (média do treino) e `aleatorio` (uniforme, semente 0).
  - `main(data_dir, out_dir, arch="cnn"|"resnet", epochs=30, batch=64, lr=1e-3)`: divisão 80/10/10 por shard, Adam, `CosineAnnealingLR`, salva `checkpoints/<arch>.pt` (state_dict + `stats`) e `checkpoints/<arch>_mae.json` com o MAE por parâmetro no teste.
- [ ] Testes: `baseline_mae` da média, com alvos constantes, dá zero; uma época em 8 exemplos sintéticos roda e grava o `.pt`.
- [ ] Rodar no Colab com o dataset completo; anotar o MAE por parâmetro contra as linhas de base.
- [ ] Commit (sem os pesos).

### Task 7: `match`: da música ao preset

**Arquivos:**
- Criar: `timbrepro/match.py`
- Modificar: `timbrepro/__main__.py`, `requirements.txt` (adicionar `demucs`)
- Teste: `tests/test_match.py`

**Interfaces:**
- Consome: `log_mel`, `TimbreCNN`/`make_resnet18`, o `.pt` da Task 6, `params_from_dict`/`load_preset`.
- Produz:
  - `separate_guitar(path: Path, work_dir: Path) -> np.ndarray`: roda `python -m demucs -n htdemucs_6s --two-stems guitar -o work_dir path` via `subprocess`, lê `guitar.wav` com `read_mono`.
  - `pick_windows(audio: np.ndarray, n: int = 8, sr: int = SAMPLE_RATE) -> list[np.ndarray]`: janelas de 3 s sem sobreposição, as `n` de maior RMS, na ordem do tempo; áudio menor que 3 s vira uma janela completada com zeros.
  - `predict_params(model, stats, windows) -> np.ndarray`: mediana por parâmetro das previsões, em `(11,)`.
  - `write_preset(params: np.ndarray, path: Path, name: str) -> None`: `clip` em [0, 1], arredonda para 3 casas, grava `{"name", "params": {nome: valor}}`.
  - CLI: `python -m timbrepro match musica.wav --output presets/musica.json [--no-separate] [--model checkpoints/cnn.pt] [--live --device 12,10]`. Imprime os 11 valores como barras de texto.
- [ ] Testes: `pick_windows` com 20 s de áudio onde só 2–5 s e 10–13 s têm som → as duas primeiras janelas devolvidas são essas; 1 s de áudio → 1 janela de 3 s; `write_preset` com um valor 1,0000001 grava 1.0 e o arquivo passa por `load_preset`; `predict_params` com um modelo falso que devolve linhas conhecidas → a mediana.
- [ ] Rodar, implementar, rodar, commit.
- [ ] Manual: `match` numa música real com `--live`.

### Task 8: Avaliação no nível do som

**Arquivos:**
- Criar: `timbrepro/evaluate.py`
- Teste: `tests/test_evaluate.py`

**Interfaces:**
- Produz:
  - `mel_l1(a: np.ndarray, b: np.ndarray) -> float`: média de `|log_mel(a) - log_mel(b)|`.
  - `mrstft(a, b) -> float`: soma, para `n_fft` em (512, 1024, 2048), de convergência espectral + L1 do log da magnitude.
  - `evaluate(model_path, data_dir, di_dir, n: int = 200) -> dict`: para `n` exemplos do teste, renderiza o mesmo DI com o preset real, o previsto, o médio e um aleatório; devolve médias de `mel_l1` e `mrstft` por método.
  - CLI: `python -m timbrepro evaluate --model checkpoints/cnn.pt --data data/dataset --di-dir data/di`.
- [ ] Testes: distância de um sinal para ele mesmo é 0; um sinal com overdrive está mais longe do seco que um sinal com EQ leve.
- [ ] Rodar, implementar, rodar, commit.

### Task 9 (opcional, só se houver folga em 10/11): refinamento por busca

**Arquivos:** modificar `timbrepro/match.py`, `tests/test_match.py`.

**Interfaces:**
- `refine(params0, reference, di, steps: int = 40) -> np.ndarray`: `scipy.optimize.minimize(method="Nelder-Mead")` sobre `mel_l1` do espectro médio no tempo entre `render_example(di, p)` e a referência, com `p` limitado a [0, 1] por `clip`. Flag `--refine` no `match`.
- [ ] Teste: partindo de um preset errado em `drive_gain` (0,2 contra 0,8 real) num exemplo sintético, o refinamento reduz o erro de `drive_gain`.
