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

O script `train.py` é responsável por treinar o agente. Você pode customizar os hiperparâmetros (como `LAMBDA`, `ALPHA`, `NUM_EPISODES`, etc.) diretamente no início do arquivo.

```bash
# Execute o script de treinamento
python train.py
```

Ao final, dois arquivos serão gerados, nomeados com base no valor de `LAMBDA` (ex: `treino.npy` e `treino.txt`):
- **`.npy`**: Contém a matriz Q aprendida pelo agente e todos os parâmetros de treinamento.
- **`.txt`**: Um arquivo de log com o progresso do treinamento (recompensa a cada 100 episódios).

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

Uma janela do Gymnasium será aberta, mostrando o agente em ação. Ao final, um arquivo de log de teste (ex: `treino_TestLog.txt`) será criado com a recompensa obtida em cada um dos 100 episódios de teste e a média final.

## Parâmetros de Treinamento (em `train.py`)

Você pode ajustar os seguintes parâmetros no início do script `train.py` para experimentar diferentes configurações:

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

## Arquivos do Projeto

- **`train.py`**: Script principal para executar o treinamento do agente.
- **`test.py`**: Script para carregar um agente treinado e avaliá-lo visualmente.
- **`q_lambda.py`**: Contém a classe `QLambdaCausal` que implementa o algoritmo de aprendizado.
- **`custom_termination_wrapper.py`**: Wrapper do Gymnasium para customizar as condições de término do ambiente (limites de ângulo, posição, etc.).
