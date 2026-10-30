# TimbrePro

Pedaleira de guitarra em software que roda em tempo real e tem o timbre ajustado por IA sob
demanda. O guitarrista pede um timbre ("soa como a guitarra desta música"), a IA define os
parâmetros dos efeitos em poucos segundos, e a partir daí ele toca ao vivo com esse preset.

O sistema tem duas partes independentes, e só a primeira precisa ser rápida:

- **Motor DSP em tempo real:** cadeia de efeitos própria, com meta de latência abaixo de 15 ms.
- **Módulo de IA offline:** uma CNN sobre espectrogramas mel que, a partir de um áudio de
  referência, estima os parâmetros da cadeia e entrega um preset JSON.

## Estado atual

Este é o protótipo inicial: só o motor DSP existe por enquanto. **Ainda não há IA no repositório.**
A interface com ela já está definida: o vetor de 11 parâmetros de [params.py](timbrepro/params.py).

- [x] Os 4 efeitos, processando em blocos e guardando estado entre eles
- [x] Troca de parâmetros sem clique
- [x] Presets em JSON, trocados com o áudio rodando
- [x] Render offline e stream ao vivo usando a mesma cadeia
- [ ] Tocar com guitarra e interface de áudio de verdade (o stream só foi testado com a saída muda)
- [ ] Medir a latência de ponta a ponta (loopback físico) e chegar a menos de 15 ms
- [ ] Gravar os primeiros trechos de guitarra limpa em DI

## Como rodar

Requer Python 3.10 ou mais novo.

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Os comandos abaixo rodam a partir da raiz do repositório.

```powershell
# Aplica um preset. Sem --input, usa um riff sintético, então não precisa de gravação.
python -m timbrepro render --preset presets/lead.json --output lead.wav
python -m timbrepro render --input minha_gravacao.wav --preset presets/crunch.json --output crunch.wav

# Mede o tempo de processamento por bloco. Não precisa de placa de som.
python -m timbrepro bench
python -m timbrepro bench --block 128

# Toca ao vivo. Com o terminal aberto, digite o caminho de outro preset para trocar.
python -m sounddevice                      # lista os dispositivos
python -m timbrepro live --device 12,10    # entrada,saída
python -m timbrepro live --device 12,10 --bypass   # passthrough, sem efeitos

# Testes
python -m pytest
```

No modo ao vivo a guitarra precisa estar na **entrada 1** da interface. No Windows, a menor
latência vem do ASIO, que fica desligado por padrão; ligue antes de listar os dispositivos:

```powershell
$env:SD_ENABLE_ASIO = "1"
```

## A cadeia de efeitos

A ordem é a de uma pedaleira típica. Os 11 parâmetros, normalizados de 0 a 1, são o vetor que a
IA vai prever.

| Ordem | Efeito | Técnica | Parâmetros |
| ----- | ------ | ------- | ---------- |
| 1 | Overdrive | Pré-ganho, `tanh` assimétrico, passa-baixas de tom | `drive_gain`, `drive_tone`, `drive_level` |
| 2 | EQ de 3 bandas | Biquads: low shelf, peaking, high shelf (0,5 = 0 dB) | `eq_low`, `eq_mid`, `eq_high` |
| 3 | Delay | Buffer circular, realimentação filtrada | `delay_time`, `delay_feedback`, `delay_mix` |
| 4 | Reverb | Freeverb: 8 pentes e 4 passa-tudo | `reverb_size`, `reverb_mix` |

A mesma classe `FxChain` serve ao tempo real e à geração do dataset, para o modelo aprender
exatamente os efeitos que vão tocar. Um teste garante que a saída não depende de como o áudio é
cortado em blocos.

## Estrutura

```text
timbrepro/
  params.py     os 11 parâmetros e a leitura de presets
  effects.py    os 4 efeitos
  chain.py      a cadeia completa e a suavização de parâmetros
  live.py       stream em tempo real
  synth.py      guitarra sintética, enquanto não há gravações DI
  __main__.py   linha de comando
presets/        presets JSON
tests/          testes
```

## Primeiras medições

Medido num notebook com Windows 11, sem otimização nem prioridade de tempo real. Os números variam
bastante entre execuções; rode `bench` na sua máquina.

| Bloco | Limite por bloco | Média | p99 | Pior bloco |
| ----- | ---------------- | ----- | --- | ---------- |
| 256 amostras | 5,33 ms | 0,5 a 1,2 ms | 1,9 a 3,8 ms | 3,7 a 6,3 ms |
| 128 amostras | 2,67 ms | 0,5 a 1,9 ms | 2,0 a 7,2 ms | 3,3 a 13,7 ms |

- **O padrão é 256 amostras:** com 128, o p99 às vezes passa do limite. Os picos vêm em boa parte do
  sistema: um trabalho numpy fixo, sem nada da cadeia, já tem p99 6 vezes maior que a mediana.
- **Grande parte do custo é por chamada, não por amostra:** o tempo médio quase não muda entre 128 e
  256. Se precisarmos descer para 128, o caminho é `numba` nos efeitos.
- **A latência ainda está longe da meta:** o áudio onboard (Realtek) reportou 112 ms em MME e 49 ms em
  WASAPI, com 256 amostras. Esses valores vêm do driver, não de um loopback. A meta de 15 ms depende
  de uma interface com ASIO, que ainda não foi testada.

## Próximos passos

| Semana | Checkpoint |
| ------ | ---------- |
| 1 | Passthrough com latência medida; 4 efeitos funcionando offline; primeiras gravações DI |
| 2 | Gerador de dataset rodando; efeitos em tempo real sem estalos |
| 3 | Primeira CNN treinada e comparada com as linhas de base; troca de preset ao vivo |
| 4 | Fluxo completo (música → separação → preset → tocar) com 3 músicas reais |
| 5 | Teste de escuta, relatório, ensaio e apresentação |

O modelo só vai reproduzir timbres alcançáveis por esta cadeia: amplificadores reais e outros pedais
ficam fora, e o objetivo é a melhor aproximação possível com os nossos efeitos.
