# ArboIA

Sistema de previsão de arboviroses com base em séries temporais, dados históricos semanais e modelos de aprendizado de máquina treinados por doença.

O projeto oferece duas formas de uso:

- **Interface web em Streamlit** para análise visual e projeção de tendência.
- **Interface em terminal** para previsões rápidas por data-alvo.

Ele foi desenhado para trabalhar com três doenças:

- Dengue
- Zika
- Chikungunya

## Visão Geral

O fluxo principal do projeto é:

1. Ler os dados históricos de cada doença em `data/`.
2. Preparar variáveis de entrada com base em:
   - semana epidemiológica
   - sazonalidade mensal
   - defasagens de casos (`lag_1`, `lag_2`, `lag_4`)
   - média móvel recente
   - temperatura média, quando disponível
3. Treinar um modelo `XGBRegressor` para cada doença.
4. Salvar o modelo e a lista de features em `modelos_salvos/`.
5. Usar esses artefatos para gerar previsões no terminal ou no dashboard.

## Estrutura do Projeto

```text
.
├── app.py              # Interface web com Streamlit
├── main.py             # Interface em terminal
├── treinamento.py      # Treinamento e validação dos modelos
├── data/               # Séries históricas usadas no treino
├── debug/              # Séries auxiliares para validação
└── modelos_salvos/     # Modelos e features serializados
```

## Como Funciona

### Treinamento

O script `treinamento.py` lê todos os CSVs de `data/`, prepara as variáveis e treina um modelo por doença usando `XGBRegressor`.

Durante a preparação dos dados, o sistema cria:

- `semana`
- `mes_sin`
- `lag_1`
- `lag_2`
- `lag_4`
- `media_movel_4s`
- `temp_shift`, quando há temperatura média disponível

Os modelos são salvos como arquivos `.joblib` em `modelos_salvos/`.

### Previsão

As previsões usam os modelos treinados e os dados históricos mais recentes para montar a entrada correta. A lógica:

- busca o modelo correspondente à doença
- carrega a lista de features esperadas
- monta um vetor de entrada com base no histórico recente
- executa a predição
- converte o valor de volta para a escala original

## Requisitos

O projeto usa Python e as bibliotecas abaixo:

- `pandas`
- `numpy`
- `joblib`
- `colorama`
- `xgboost`
- `scikit-learn`
- `tqdm`
- `matplotlib`
- `streamlit`
- `plotly`

## Instalação

```bash
pip install pandas numpy joblib colorama xgboost scikit-learn tqdm matplotlib streamlit plotly
```

Se preferir, crie um ambiente virtual antes de instalar:

```bash
python -m venv .venv
source .venv/bin/activate
```

## Como Treinar os Modelos

Execute:

```bash
python treinamento.py
```

Esse comando:

- lê os dados de `data/`
- treina os modelos
- salva os artefatos em `modelos_salvos/`
- valida, quando existir, com os dados de `debug/`

## Como Usar no Terminal

```bash
python main.py
```

No terminal, você poderá:

- escolher a doença
- informar uma data alvo no formato `AAAA-MM-DD`
- receber a estimativa de casos para a data informada

## Como Usar no Dashboard Web

```bash
streamlit run app.py
```

O painel permite:

- selecionar a doença
- escolher a data de previsão
- visualizar histórico real e projeção
- acompanhar a tendência em gráfico

## Arquivos de Dados

Os datasets ficam em `data/` e devem seguir o padrão:

- `dengue.csv`
- `zika.csv`
- `chikungunya.csv`

O sistema identifica automaticamente:

- a coluna de data
- a coluna de casos (`casos` ou `casos_est`)
- a coluna `tempmed`, se existir

## Artefatos Gerados

Após o treinamento, o projeto salva:

- `modelo_DENGUE.joblib`
- `modelo_ZIKA.joblib`
- `modelo_CHIKUNGUNYA.joblib`
- `features_DENGUE.joblib`
- `features_ZIKA.joblib`
- `features_CHIKUNGUNYA.joblib`

Esses arquivos são usados pela interface web e pelo terminal.

## Observações Importantes

- As previsões dependem de histórico suficiente para calcular as defasagens.
- O modelo trabalha com dados semanais.
- Se o modelo correspondente ainda não tiver sido treinado, o sistema exibirá erro pedindo o treinamento primeiro.
- O projeto foi pensado para uso local com os arquivos já presentes no repositório.

## Limitações

Este projeto é um sistema de apoio à análise, não um diagnóstico clínico.

As estimativas dependem da qualidade do histórico, da consistência dos CSVs e da disponibilidade dos campos esperados.

## Próximos Passos

Se quiser evoluir o projeto, os caminhos mais naturais são:

- adicionar tratamento de erros mais explícito para CSVs incompletos
- criar `requirements.txt`
- padronizar métricas de avaliação por doença
- incluir fontes e atualização automatizada dos dados
- separar a lógica de previsão em um módulo reutilizável

