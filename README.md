Grupo:
- Pedro Ulisses de Lima Quadros
- Iago Jacob de Souza Ramos
- Leonardo Fernandes Trevilato
- Rhuan dos Santos Vicente

## Sobre o Projeto

Este projeto utiliza o algoritmo de aprendizado por reforço **Q(λ) Causal** para treinar um agente a resolver o problema clássico do **Pêndulo Invertido** (`InvertedPendulum-v5`) do Gymnasium.

O objetivo é equilibrar um poste na vertical em cima de um carrinho que pode se mover horizontalmente. O agente aprende uma política ótima, ou seja, uma estratégia para escolher ações (mover para a esquerda ou direita) que maximize a recompensa total ao longo do tempo.

## Como Usar

Siga os passos abaixo para configurar o ambiente, treinar e testar o agente.

### 1. Configuração do Ambiente

É altamente recomendado usar um ambiente virtual (`venv`) para isolar as dependências do projeto.

```bash
# 1. Crie um ambiente virtual na pasta do projeto
python -m venv venv

# 2. Ative o ambiente virtual
# No Windows:
.\venv\Scripts\activate
# No macOS/Linux:
source venv/bin/activate

# 3. Instale as bibliotecas necessárias
pip install -r requirements.txt
```

### 2. Treinando o Agente

O script `train.py` é responsável por treinar o agente. Você pode customizar os hiperparâmetros (como `LAMBDA`, `ALPHA`, `NUM_EPISODES`, etc.) nos arquivos da pasta `configs` e selecionar o arquivo de configuração no começo de `train.py` alterando a variável `CONFIG_NAME`.

```bash
# Execute o script de treinamento
python train.py
```

**Durante o treinamento**, você verá logs no console a cada 50 episódios mostrando:
- **Episódio atual** e progresso
- **Taxa de Sucesso**: Porcentagem de episódios bem-sucedidos (recompensa ≥ 900) nos últimos 300 episódios
- **Recompensa**: Recompensa total do episódio atual
- **Epsilon**: Nível atual de exploração
- **Tempo decorrido**

Exemplo de log:
```
Episódio: 150/4000, Taxa Sucesso: 12.67%, Recompensa: 456.78, Epsilon: 0.9384, Tempo: 45s
```

Ao final, dois arquivos serão gerados na pasta correspondente:
- **`npy/[nome].npy`**: Contém a matriz Q aprendida e todos os parâmetros de treinamento.
- **`trainLog/[nome].txt`**: Log detalhado com progresso do treinamento.

### Early Stopping (Parada Antecipada)

O treinamento pode parar automaticamente antes de completar todos os episódios em duas situações:

#### **Critério de Sucesso**
- **Condição**: Taxa de sucesso ≥ 98%
- **Significado**: 294 dos últimos 300 episódios tiveram recompensa ≥ 900
- **Objetivo**: Evita treinamento desnecessário quando o agente já dominou a tarefa

#### **Critério de Platô**
- **Condição**: Performance estável por 500 episódios com pouca variação (≤ 10) e média ≥ 810
- **Significado**: O agente parou de melhorar significativamente
- **Objetivo**: Economiza tempo quando não há mais progresso esperado


### 3. Testando o Agente

Após o treinamento, use o script `test.py` para avaliar o desempenho do agente em um ambiente com visualização.

1.  **Abra o arquivo `test.py`** e altere a variável `FILENAME_BASE` para corresponder ao nome do modelo que você deseja testar (sem a extensão `.npy`).

    ```python
    # Exemplo em test.py
    FILENAME_BASE = "treino" 
    ```

2.  Execute o script de teste no terminal:
    ```bash
    python test.py
    ```

Uma janela do Gymnasium será aberta, mostrando o agente em ação. O teste executa **20 episódios** sem exploração (epsilon=0) para avaliar a performance pura do agente. Ao final, um arquivo de log de teste será criado com a recompensa de cada episódio e a média final.

## Parâmetros de Treinamento (em `train.py`)

Você pode ajustar os seguintes parâmetros nos arquivps de configuração na pasta `configs`para experimentar diferentes configurações:

### Parâmetros Físicos e de Ambiente
- **`GRAVITY`**: Altera a força da gravidade no ambiente.
- **`FORCE_MAGNITUDE`**: Magnitude da força horizontal aplicada ao carrinho.
- **`POSITION_LIMIT`**, **`ANGLE_LIMIT_RADS`**, **`VELOCITY_LIMIT`**, **`ANGULAR_VELOCITY_LIMIT`**: Limites físicos que definem uma falha no episódio. Se o agente ultrapassar qualquer um desses limites, o episódio é encerrado.

### Hiperparâmetros do Algoritmo Q(λ)
- **`ALPHA`**: Taxa de Aprendizado. Controla o tamanho do passo de atualização dos valores Q. Valores mais altos significam aprendizado mais rápido, mas podem levar à instabilidade.
- **`GAMMA`**: Fator de Desconto. Pondera a importância de recompensas futuras. Um valor próximo de 1 faz o agente se preocupar mais com o longo prazo.
- **`LAMBDA`**: Fator de Decaimento do Rastro de Elegibilidade. Determina como o crédito de uma recompensa é distribuído para os estados e ações visitados anteriormente. `λ=0` equivale ao Q-Learning de um passo.

### Parâmetros de Exploração (Epsilon-Greedy)
- **`EPSILON`**: Probabilidade inicial de o agente escolher uma ação aleatória em vez da melhor ação conhecida. Começa alto para incentivar a exploração.
- **`EPSILON_DECAY_RATE`**: Taxa pela qual `EPSILON` diminui a cada passo. Isso faz com que o agente explore menos e explore mais (explote) à medida que aprende.
- **`MIN_EPSILON`**: Valor mínimo que `EPSILON` pode atingir, garantindo que sempre haja uma pequena chance de exploração.

### Discretização do Espaço de Estados
- **`N_POSITION`**, **`N_ANGLE`**, **`N_VELOCITY`**, **`N_ANGULAR_VELOCITY`**: Número de "caixas" ou "bins" para dividir cada variável contínua do estado (posição, ângulo, velocidade e velocidade angular). Uma granularidade maior (números mais altos) permite uma representação mais precisa do estado, mas aumenta drasticamente a memória e o tempo de treinamento.

### Configurações Gerais de Treinamento
- **`NUM_EPISODES`**: O número total de episódios que o agente irá treinar.
- **`MAX_STEPS`**: O número máximo de passos (ações) que um agente pode executar em um único episódio antes que ele seja encerrado. Isso incentiva o agente a se equilibrar por mais tempo para obter uma recompensa maior.

### Configurações de Early Stopping
- **`EARLY_STOP_SUCCESS_RATE`**: Taxa de sucesso mínima (98%) para parada antecipada por sucesso.
- **`EARLY_STOP_THRESHOLD`**: Recompensa mínima (900) para considerar um episódio "bem-sucedido".
- **`EARLY_STOP_WINDOW`**: Janela de avaliação (300 episódios) para calcular a taxa de sucesso.
- **`PLATEAU_WINDOW`**: Janela (500 episódios) para detectar platô de performance.
- **`PLATEAU_TOLERANCE`**: Variação máxima (10) permitida para considerar platô.

## Arquivos do Projeto

- **`train.py`**: Script principal para executar o treinamento do agente.
- **`test.py`**: Script para carregar um agente treinado e avaliá-lo visualmente.
- **`grafico.py`**: Script para gerar gráficos da evolução do treinamento.
- **`q_lambda.py`**: Contém a classe `QLambdaCausal` que implementa o algoritmo de aprendizado.
- **`custom_termination_wrapper.py`**: Wrapper do Gymnasium para customizar as condições de término do ambiente.
- **`npy/`**: Pasta contendo os modelos treinados (arquivos `.npy`).
- **`trainLog/`**: Pasta contendo os logs de treinamento (arquivos `.txt`).
- **`requirements.txt`**: Lista de dependências do projeto.

### 4. Visualizando os Resultados

Durante o treinamento, o script `grafico.py` é executado para gerar gráficos da evolução do treinamento:

O script irá:
- **Ler os logs** de treinamento (arquivos `.txt`)
- **Gerar gráficos** mostrando a evolução da taxa de sucesso
- **Incluir estatísticas** como máximo, média geral e média final
- **Mostrar parâmetros** usados no treinamento (λ, α, γ)

O gráfico inclui:
- **Linha azul**: Taxa de sucesso por episódio
- **Linha vermelha**: Média móvel (suavização)
- **Linhas de referência**: 50% e 90% de sucesso
- **Estatísticas**: Performance máxima, média geral e final