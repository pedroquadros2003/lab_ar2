import gymnasium as gym
import time
import numpy as np
from custom_termination_wrapper import CustomTerminationWrapper
from q_lambda import QLambdaCausal


# --- Parâmetros Físicos Configuráveis ---
GRAVITY = 5.0
FORCE_MAGNITUDE = 0.75

# --- Restrições Físicas Impostas ao Problema ---

POSITION_LIMIT = 1
ANGLE_LIMIT_RADS = 0.5
VELOCITY_LIMIT = 3
ANGULAR_VELOCITY_LIMIT = 3

# --- Hiperparâmetros do Algoritmo Q(λ) ---

ALPHA = 0.05  # Taxa de aprendizado (learning rate)
GAMMA = 0.95 # Fator de desconto para recompensas futuras
LAMBDA = 0.8 # Fator de decaimento para os rastros de elegibilidade

# --- Parâmetros de Exploração (Epsilon-Greedy) ---

EPSILON = 1.0
EPSILON_DECAY_RATE = 0.00001
MIN_EPSILON = 0.01

# --- Parâmetros da Discretização do Espaço de Estados ---
# Define em quantas "caixas" cada variável contínua será dividida.
N_POSITION = 6
N_ANGLE = 10
N_VELOCITY = 6
N_ANGULAR_VELOCITY = 10
STATE_DIMS = (N_POSITION, N_ANGLE, N_VELOCITY, N_ANGULAR_VELOCITY)

# --- Parâmetros de Treinamento ---
NUM_EPISODES = 30000

# --- Parâmetros de Saída ---
FILENAME_BASE = f"treino_lambda_{LAMBDA}"
OUTPUT_FILENAME = f"{FILENAME_BASE}.npy"   
LOG_FILENAME = f"{FILENAME_BASE}.txt"

# --- Instanciação do Agente de IA ---
q_agent = QLambdaCausal(
    dims=STATE_DIMS,
    alpha=ALPHA,
    lambda_=LAMBDA,
    gamma=GAMMA,
    pos_limit=POSITION_LIMIT,
    angle_limit=ANGLE_LIMIT_RADS,
    vel_limit=VELOCITY_LIMIT,
    ang_vel_limit=ANGULAR_VELOCITY_LIMIT,
    num_actions=2 # Duas ações: esquerda e direita
)

# 1. Crie o ambiente. Use 'human' para ver o agente ou 'rgb_array' para treinar mais rápido.
env = gym.make('InvertedPendulum-v5', render_mode='rgb_array')

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
observation, info = env.reset(seed=42) # Inicializamos uma seed para tornar o treino reprodutível
epsilon = EPSILON

running = True
with open(LOG_FILENAME, 'w') as log_file:
    try:
        # --- Cria e escreve o cabeçalho no arquivo de log ---
        header = f"""
======================================================================
               RELATÓRIO DE TREINAMENTO - PÊNDULO INVERTIDO
======================================================================

--- Hiperparâmetros do Algoritmo Q(λ) ---
Taxa de Aprendizado (ALPHA): {ALPHA}
Fator de Desconto (GAMMA):   {GAMMA}
Fator de Decaimento (LAMBDA):  {LAMBDA}

--- Parâmetros de Exploração (Epsilon-Greedy) ---
Epsilon Inicial:      {EPSILON}
Taxa de Decaimento:   {EPSILON_DECAY_RATE}
Epsilon Mínimo:       {MIN_EPSILON}

--- Discretização do Espaço de Estados ---
Posição:     {N_POSITION} bins
Ângulo:      {N_ANGLE} bins
Velocidade:  {N_VELOCITY} bins
Vel. Angular:{N_ANGULAR_VELOCITY} bins

--- Configurações Gerais ---
Episódios de Treino: {NUM_EPISODES}
======================================================================\n\n"""
        log_file.write(header)

        for episode in range(NUM_EPISODES):
            # Reinicia o ambiente para cada episódio
            prev_observation, info = env.reset()
            terminated = False
            truncated = False
            total_reward = 0

            while not terminated and not truncated:
                # 3. O agente escolhe uma ação com base na política epsilon-greedy
                prev_state_idx = q_agent.convert2state(prev_observation)
                action_idx = q_agent.choose_action(prev_state_idx, epsilon)

                # Mapeia o índice da ação para a força a ser aplicada
                # Ação 0: Esquerda, Ação 1: Direita
                action_force = [-FORCE_MAGNITUDE] if action_idx == 0 else [FORCE_MAGNITUDE]
                
                # 4. Execute a ação.
                observation, reward, terminated, truncated, info = env.step(action_force)
                total_reward += reward
                
                # 5. Atualiza o agente de IA com a transição
                current_state_idx = q_agent.convert2state(observation)
                q_agent.update(prev_state_idx, action_idx, reward, current_state_idx)
                
                # Guarda a observação atual para o próximo passo
                prev_observation = observation

                # Decai o epsilon para reduzir a exploração ao longo do tempo
                epsilon = max(MIN_EPSILON, epsilon - EPSILON_DECAY_RATE)

            if (episode + 1) % 100 == 0:
                log_message = f"Episódio: {episode + 1}/{NUM_EPISODES}, Recompensa Total: {total_reward:.2f}, Epsilon: {epsilon:.4f}"
                print(log_message)
                log_file.write(log_message + '\n')

    finally:
        # 5. Garante que o ambiente será fechado ao final.
        env.close()
        final_message1 = "Treinamento concluído. Salvando os valores Q..."
        print(final_message1)
        log_file.write(final_message1 + '\n')

        # 6. Agrupa a matriz Q e os parâmetros para salvar em um único arquivo.
        data_to_save = {
            'q_matrix': q_agent.q_matrix,
            'params': {
                'GRAVITY': GRAVITY,
                'FORCE_MAGNITUDE': FORCE_MAGNITUDE,
                'POSITION_LIMIT': POSITION_LIMIT,
                'ANGLE_LIMIT_RADS': ANGLE_LIMIT_RADS,
                'VELOCITY_LIMIT': VELOCITY_LIMIT,
                'ANGULAR_VELOCITY_LIMIT': ANGULAR_VELOCITY_LIMIT,
                'N_POSITION': N_POSITION,
                'N_ANGLE': N_ANGLE,
                'N_VELOCITY': N_VELOCITY,
                'N_ANGULAR_VELOCITY': N_ANGULAR_VELOCITY,
                'LAMBDA': LAMBDA
            }
        }
        
        # Salva o dicionário em um arquivo binário NumPy.
        np.save(OUTPUT_FILENAME, data_to_save)
        
        final_message2 = f"Matriz Q salva em '{OUTPUT_FILENAME}'."
        print(final_message2)
        log_file.write(final_message2 + '\n')
