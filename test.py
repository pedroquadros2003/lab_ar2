import gymnasium as gym
import time
import sys
import numpy as np
from custom_termination_wrapper import CustomTerminationWrapper
from q_lambda import QLambdaCausal

# --- Parâmetros de Entrada e Saída ---
# Especifique o arquivo .npy do modelo treinado que você quer testar.
FILENAME_BASE = "treino_Qlambda" 
INPUT_Q_VALUES_FILENAME = f"npy/{FILENAME_BASE}.npy"
TEST_LOG_FILENAME = f"trainLog/{FILENAME_BASE}_TestLog.txt"

# --- Parâmetros de Teste ---
NUM_TEST_EPISODES = 20 # Número de episódios para rodar o teste
FPS = 10000 # Para visualização
RENDER_MODE = 'human' # Use 'human' para ver o agente ou 'rgb_array' para rodar sem visualização.

# --- Carregamento do Modelo e Parâmetros ---
try:
    print(f"Carregando dados do arquivo: '{INPUT_Q_VALUES_FILENAME}'...")
    # allow_pickle=True é necessário para carregar dicionários salvos com np.save
    saved_data = np.load(INPUT_Q_VALUES_FILENAME, allow_pickle=True).item()
    
    q_matrix = saved_data['q_matrix']
    params = saved_data['params']
    
    print("Dados carregados com sucesso.")
except FileNotFoundError:
    print(f"ERRO: Arquivo '{INPUT_Q_VALUES_FILENAME}' não encontrado. Verifique o nome e o caminho do arquivo.")
    sys.exit()
except KeyError as e:
    print(f"ERRO: O arquivo .npy não contém a chave esperada: {e}. Ele foi gerado pelo script de treino atualizado?")
    sys.exit()

# --- Extração dos Parâmetros Carregados ---
GRAVITY = params['GRAVITY']
FORCE_MAGNITUDE = params['FORCE_MAGNITUDE']
POSITION_LIMIT = params['POSITION_LIMIT']
ANGLE_LIMIT_RADS = params['ANGLE_LIMIT_RADS']
VELOCITY_LIMIT = params['VELOCITY_LIMIT']
ANGULAR_VELOCITY_LIMIT = params['ANGULAR_VELOCITY_LIMIT']
STATE_DIMS = (params['N_POSITION'], params['N_ANGLE'], params['N_VELOCITY'], params['N_ANGULAR_VELOCITY'])

# --- Instanciação do Agente de IA ---
q_agent = QLambdaCausal(
    dims=STATE_DIMS,
    alpha=0, lambda_=0, gamma=0, # Hiperparâmetros não são usados no teste
    pos_limit=params['POSITION_LIMIT'],
    angle_limit=params['ANGLE_LIMIT_RADS'],
    vel_limit=params['VELOCITY_LIMIT'],
    ang_vel_limit=params['ANGULAR_VELOCITY_LIMIT'],
    num_actions=2 
)
q_agent.q_matrix = q_matrix

# 1. Crie o ambiente.
env = gym.make('InvertedPendulum-v5', render_mode=RENDER_MODE)

# Modifica a gravidade e as condições de término para corresponder ao treino.
env.unwrapped.model.opt.gravity[2] = -GRAVITY
env = CustomTerminationWrapper(
    env, 
    angle_limit=ANGLE_LIMIT_RADS, 
    pos_limit=POSITION_LIMIT, 
    vel_limit=VELOCITY_LIMIT, 
    ang_vel_limit=ANGULAR_VELOCITY_LIMIT
)

total_rewards_list = []

with open(TEST_LOG_FILENAME, 'w', encoding='utf-8') as log_file:
    try:
        header = f"""
======================================================================
               RELATÓRIO DE TESTE - PÊNDULO INVERTIDO
======================================================================
Modelo Carregado: {INPUT_Q_VALUES_FILENAME}
Número de Episódios de Teste: {NUM_TEST_EPISODES}
======================================================================\n\n"""
        log_file.write(header)
        print("\nIniciando o teste...")
        
        # Semeia o ambiente para garantir que os episódios de teste sejam reprodutíveis
        observation, info = env.reset(seed=42)
        
        for episode in range(NUM_TEST_EPISODES):
            terminated = False
            truncated = False
            total_reward = 0

            while not terminated and not truncated:
                # 1. Converte a observação para o estado discreto
                state_idx = q_agent.convert2state(observation)
                
                # 2. O agente escolhe a MELHOR ação (epsilon=0)
                action_idx = q_agent.choose_action(state_idx, epsilon=0)

                # Mapeia o índice da ação para a força
                action_force = [-FORCE_MAGNITUDE] if action_idx == 0 else [FORCE_MAGNITUDE]
                
                # 3. Executa a ação no ambiente
                observation, reward, terminated, truncated, info = env.step(action_force)
                total_reward += reward
                
                # Pausa para visualização
                time.sleep(1 / FPS)
            
            total_rewards_list.append(total_reward)
            log_message = f"Episodio de Teste: {episode + 1}/{NUM_TEST_EPISODES}, Recompensa Total: {total_reward:.2f}"
            print(log_message)
            log_file.write(log_message + '\n')
            
            # Reinicia o ambiente para o próximo episódio
            observation, info = env.reset()

    finally:
        env.close()
        
        # Calcula e salva a média das recompensas
        if total_rewards_list:
            average_reward = np.mean(total_rewards_list)
            summary_message = f"\nRecompensa Media em {NUM_TEST_EPISODES} episodios: {average_reward:.2f}"
        else:
            summary_message = "\nNenhum episodio de teste foi completado."

        print("\nTeste concluido.")
        print(summary_message)
        log_file.write(summary_message + '\n')
        print(f"Log de teste salvo em '{TEST_LOG_FILENAME}'.")
