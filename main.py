import gymnasium as gym
import time
import keyboard
import numpy as np
from custom_termination_wrapper import CustomTerminationWrapper
from td_lambda import TDLambdaCausal


# --- Parâmetros Físicos Configuráveis ---

FPS = 20 
GRAVITY = 5.0
FORCE_MAGNITUDE = 0.75

# --- Restrições Físicas Impostas ao Problema ---

POSITION_LIMIT = 1
ANGLE_LIMIT_RADS = 0.5
VELOCITY_LIMIT = 3
ANGULAR_VELOCITY_LIMIT = 3

# --- Hiperparâmetros do Algoritmo TD Causal ---

ALPHA = 0.1  # Taxa de aprendizado (learning rate)
GAMMA = 0.99 # Fator de desconto para recompensas futuras
LAMBDA = 0.9 # Fator de decaimento para os rastros de elegibilidade

# --- Parâmetros da Discretização do Espaço de Estados ---
# Define em quantas "caixas" cada variável contínua será dividida.
N_POSITION = 20
N_ANGLE = 20
N_VELOCITY = 20
N_ANGULAR_VELOCITY = 20
STATE_DIMS = (N_POSITION, N_ANGLE, N_VELOCITY, N_ANGULAR_VELOCITY)

# --- Parâmetros de Saída ---
OUTPUT_FILENAME = "cost_values.txt"


# --- Instanciação do Agente de IA ---
td_agent = TDLambdaCausal(
    dims=STATE_DIMS,
    alpha=ALPHA,
    lambda_=LAMBDA,
    gamma=GAMMA,
    pos_limit=POSITION_LIMIT,
    angle_limit=ANGLE_LIMIT_RADS,
    vel_limit=VELOCITY_LIMIT,
    ang_vel_limit=ANGULAR_VELOCITY_LIMIT
)

# 1. Crie o ambiente com render_mode='human' para que possamos vê-lo.
env = gym.make('InvertedPendulum-v5', render_mode='human')

# Modifica a gravidade diretamente no modelo da simulação após a criação.
# A gravidade atua no eixo Z (índice 2) e deve ser negativa para puxar para baixo.
env.unwrapped.model.opt.gravity[2] = -GRAVITY
env = CustomTerminationWrapper(
    env, 
    angle_limit=ANGLE_LIMIT_RADS, 
    pos_limit=POSITION_LIMIT, 
    vel_limit=VELOCITY_LIMIT, 
    ang_vel_limit=ANGULAR_VELOCITY_LIMIT
)

# 2. Reinicie o ambiente para obter o estado inicial.
prev_observation, info = env.reset(seed=42) # Usar uma seed é bom para ter inícios reprodutíveis

running = True
try:
    while running:
        # 3. Determine a ação com base nas teclas pressionadas.
        action = [0.0] # Ação padrão: nenhuma força
        
        if keyboard.is_pressed('z'):
            action = [-FORCE_MAGNITUDE] # Aplica força para a esquerda
        elif keyboard.is_pressed('x'):
            action = [FORCE_MAGNITUDE] # Aplica força para a direita
        
        # Pressione 'esc' para sair do loop.
        if keyboard.is_pressed('esc'):
            running = False
        
        # 4. Execute a ação. O ambiente será renderizado automaticamente a cada passo.
        observation, reward, terminated, truncated, info = env.step(action)
        
        # 5. Atualiza o agente de IA com a transição
        prev_state_idx = td_agent.convert2state(prev_observation)
        current_state_idx = td_agent.convert2state(observation)
        td_agent.update(prev_state_idx, current_state_idx, reward)
        
        # Guarda a observação atual para o próximo passo
        prev_observation = observation

        # Se o episódio terminar (pêndulo caiu, etc.), reinicie-o.
        if terminated or truncated:
            observation, info = env.reset()

        # Adiciona uma pausa para controlar a velocidade da simulação (FPS)
        time.sleep(1 / FPS)

finally:
    # 5. Garante que o ambiente será fechado ao final.
    env.close()
    print("Ambiente fechado. Salvando os valores de custo...")

    # 6. Salva a matriz de custo em um arquivo de texto.
    with open(OUTPUT_FILENAME, 'w') as f:
        # Itera por cada índice e valor na matriz de custo
        for index, value in np.ndenumerate(td_agent.cost_matrix):
            # Escreve a posição (índice) e o valor V no arquivo
            f.write(f"{index}: {value}\n")
    
    print(f"Valores de custo salvos em '{OUTPUT_FILENAME}'.")




'''
posição em (m) do carrinho (esq é negativo tbm) | ângulo vertical do pêndulo  (para a esq é negativo)
|  velocidade linear do carrinho | velocidade angular do pêndulo
'''