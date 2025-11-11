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

## Arquivos do Projeto

- **`train.py`**: Script principal para executar o treinamento do agente.
- **`test.py`**: Script para carregar um agente treinado e avaliá-lo visualmente.
- **`q_lambda.py`**: Contém a classe `QLambdaCausal` que implementa o algoritmo de aprendizado.
- **`custom_termination_wrapper.py`**: Wrapper do Gymnasium para customizar as condições de término do ambiente (limites de ângulo, posição, etc.).
