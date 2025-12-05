import gymnasium as gym
import time
import os
import importlib
import numpy as np
from custom_termination_wrapper import CustomTerminationWrapper
from q_lambda import QLambdaCausal
import grafico

# --- SELEÇÃO DE CONFIGURAÇÃO ---
# Para mudar o experimento, altere o nome do arquivo na string abaixo.
CONFIG_NAME = "config2"

# Carrega dinamicamente o módulo de configuração especificado
config = importlib.import_module(f'configs.{CONFIG_NAME}')

# --- Carrega os parâmetros do arquivo de configuração ---
# Hiperparâmetros do Agente
ALPHA = config.ALPHA
GAMMA = config.GAMMA
LAMBDA = config.LAMBDA
# Parâmetros de Exploração
EPSILON = config.EPSILON
EPSILON_DECAY_RATE = config.EPSILON_DECAY_RATE
MIN_EPSILON = config.MIN_EPSILON
# Parâmetros de Arquivo
FILENAME_BASE = config.FILENAME_BASE

OUTPUT_FILENAME = f"npy/{FILENAME_BASE}.npy"   
LOG_FILENAME = f"trainLog/{FILENAME_BASE}.txt"
# Parâmetros Gerais e do Ambiente
MASTER_SEED = config.MASTER_SEED
GRAVITY = config.GRAVITY
FORCE_MAGNITUDE = config.FORCE_MAGNITUDE
POSITION_LIMIT = config.POSITION_LIMIT
ANGLE_LIMIT_RADS = config.ANGLE_LIMIT_RADS
VELOCITY_LIMIT = config.VELOCITY_LIMIT
ANGULAR_VELOCITY_LIMIT = config.ANGULAR_VELOCITY_LIMIT
N_POSITION, N_VELOCITY, N_ANGLE, N_ANGULAR_VELOCITY = config.N_POSITION, config.N_VELOCITY, config.N_ANGLE, config.N_ANGULAR_VELOCITY
NUM_EPISODES = config.NUM_EPISODES
MAX_STEPS = config.MAX_STEPS
EARLY_STOP_THRESHOLD, EARLY_STOP_WINDOW, EARLY_STOP_SUCCESS_RATE = config.EARLY_STOP_THRESHOLD, config.EARLY_STOP_WINDOW, config.EARLY_STOP_SUCCESS_RATE
MIN_EPISODES, PLATEAU_WINDOW, PLATEAU_TOLERANCE = config.MIN_EPISODES, config.PLATEAU_WINDOW, config.PLATEAU_TOLERANCE

np.random.seed(MASTER_SEED)  # Aplica a seed do NumPy
STATE_DIMS = (N_POSITION, N_ANGLE, N_VELOCITY, N_ANGULAR_VELOCITY)

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

# Crie o ambiente. Use 'human' para ver o agente ou 'rgb_array' para treinar mais rápido.
env = gym.make('InvertedPendulum-v5', render_mode=None)

# Aplica um limite de tempo (passos) para cada episódio.
env = gym.wrappers.TimeLimit(env, max_episode_steps=MAX_STEPS)

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

# Reinicie o ambiente para obter o estado inicial.
observation, info = env.reset(seed=MASTER_SEED) # Inicializamos uma seed para tornar o treino reprodutível
epsilon = EPSILON

running = True

# --- Cria os diretórios de saída se não existirem ---
for path in [OUTPUT_FILENAME, LOG_FILENAME]:
    dir_name = os.path.dirname(path)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name)

with open(LOG_FILENAME, 'w', encoding='utf-8') as log_file:
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

--- Restrições Físicas ---
Limite de Posição:    {POSITION_LIMIT}
Limite de Ângulo (rad): {ANGLE_LIMIT_RADS}
Limite de Velocidade: {VELOCITY_LIMIT}
Limite de Vel. Angular: {ANGULAR_VELOCITY_LIMIT}

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
Passos Máximos por Episódio: {MAX_STEPS}
Seed de Reprodutibilidade: {MASTER_SEED}
======================================================================\n\n"""
        log_file.write(header)

        # Inicializar tempo de treinamento e variáveis de early stopping
        start_time = time.time()
        recent_rewards = []
        performance_history = []
        
        for episode in range(NUM_EPISODES):
            # Reinicia o ambiente para cada episódio
            prev_observation, info = env.reset()
            terminated = False
            truncated = False
            total_reward = 0

            while not terminated and not truncated:
                # O agente escolhe uma ação com base na política epsilon-greedy
                prev_state_idx = q_agent.convert2state(prev_observation)
                action_idx = q_agent.choose_action(prev_state_idx, epsilon)

                # Mapeia o índice da ação para a força a ser aplicada
                # Ação 0: Esquerda, Ação 1: Direita
                action_force = [-FORCE_MAGNITUDE] if action_idx == 0 else [FORCE_MAGNITUDE]
                
                # Execute a ação.
                observation, reward, terminated, truncated, info = env.step(action_force)
                total_reward += reward
                
                # Atualiza o agente de IA com a transição
                current_state_idx = q_agent.convert2state(observation)
                q_agent.update(prev_state_idx, action_idx, reward, current_state_idx)
                
                # Guarda a observação atual para o próximo passo
                prev_observation = observation

                # Decai o epsilon para reduzir a exploração ao longo do tempo
                epsilon = max(MIN_EPSILON, epsilon - EPSILON_DECAY_RATE)

            # Armazenar recompensa para análise de early stopping
            recent_rewards.append(total_reward)
            if len(recent_rewards) > EARLY_STOP_WINDOW:
                recent_rewards.pop(0)
            
            # Função para gerar e salvar o log do episódio atual
            def log_current_episode():
                # Calcular tempo decorrido
                elapsed_time = time.time() - start_time
                episodes_per_second = (episode + 1) / elapsed_time if elapsed_time > 0 else 0
                
                # Calcular taxa de sucesso sempre - inicia em 0.00%
                if len(recent_rewards) == EARLY_STOP_WINDOW:
                    successful_recent = sum(1 for r in recent_rewards if r >= EARLY_STOP_THRESHOLD)
                    success_rate = successful_recent / EARLY_STOP_WINDOW
                elif len(recent_rewards) > 0:
                    # Calcula taxa baseada nos episódios disponíveis até agora
                    successful_recent = sum(1 for r in recent_rewards if r >= EARLY_STOP_THRESHOLD)
                    success_rate = successful_recent / len(recent_rewards)
                else:
                    # Nos primeiros episódios, taxa de sucesso é 0
                    success_rate = 0.0
                
                success_rate_str = f", Taxa Sucesso: {success_rate:.2%}"
                
                # Log simplificado - episódio, taxa de sucesso, recompensa e epsilon
                log_message = f"Episódio: {episode + 1:4}/{NUM_EPISODES}{success_rate_str:22}, Recompensa: {total_reward:4}, Epsilon: {epsilon:.5f}, Tempo: {elapsed_time:.0f}s"
                
                print(log_message)
                log_file.write(log_message + '\n')
                return success_rate
            
            # Print do episódio se for múltiplo de 50 ou se for o último episódio
            should_log_this_episode = (episode + 1) % 50 == 0 or episode + 1 == NUM_EPISODES
            if should_log_this_episode:
                current_success_rate = log_current_episode()
            
            # Critério 1: Performance consistente
            if episode >= MIN_EPISODES and len(recent_rewards) == EARLY_STOP_WINDOW:
                successful_episodes = sum(1 for r in recent_rewards if r >= EARLY_STOP_THRESHOLD)
                success_rate = successful_episodes / EARLY_STOP_WINDOW
                
                if success_rate >= EARLY_STOP_SUCCESS_RATE:
                    # Se não logamos neste episódio ainda, fazer o log antes do break
                    if not should_log_this_episode:
                        log_current_episode()
                    
                    early_stop_msg = f"""\n======================================================================
                    EARLY STOPPING - CONVERGÊNCIA DETECTADA
======================================================================
Episódio: {episode + 1}
Taxa de Sucesso: {success_rate:.2%} (últimos {EARLY_STOP_WINDOW} episódios)
Recompensa Média: {np.mean(recent_rewards):.2f}
Critério: {successful_episodes}/{EARLY_STOP_WINDOW} episódios com recompensa ≥ {EARLY_STOP_THRESHOLD}
======================================================================"""
                    print(early_stop_msg)
                    log_file.write(early_stop_msg + '\n')
                    break

            if (episode + 1) % 50 == 0:
                current_avg = np.mean(recent_rewards) if recent_rewards else 0
                performance_history.append(current_avg)
                
                # Critério 2: Detecção de platô
                if len(performance_history) >= PLATEAU_WINDOW // 100:
                    recent_performance = performance_history[-5:]  # Últimos 500 episódios
                    if len(recent_performance) >= 3:
                        performance_range = max(recent_performance) - min(recent_performance)
                        plateau_avg = np.mean(recent_performance)
                        
                        if performance_range <= PLATEAU_TOLERANCE and plateau_avg >= EARLY_STOP_THRESHOLD * 0.9:
                            # Se não logamos neste episódio ainda, fazer o log antes do break
                            if not should_log_this_episode:
                                log_current_episode()
                            
                            early_stop_msg = f"""\n======================================================================
                    EARLY STOPPING - PLATÔ DETECTADO
======================================================================
Episódio: {episode + 1}
Performance Estabilizada: {plateau_avg:.2f}
Variação nos últimos {PLATEAU_WINDOW} episódios: {performance_range:.2f}
Sem melhoria significativa detectada.
======================================================================"""
                            print(early_stop_msg)
                            log_file.write(early_stop_msg + '\n')
                            break

    finally:
        # Garante que o ambiente será fechado ao final.
        env.close()
        
        # Calcular tempo total de treinamento
        total_training_time = time.time() - start_time
        episodes_total = episode + 1
        avg_speed = episodes_total / total_training_time if total_training_time > 0 else 0
        
        final_message1 = "Treinamento concluído. Salvando os valores Q..."
        print(final_message1)
        print(f"Tempo total: {total_training_time:.0f}s, Velocidade média: {avg_speed:.2f}ep/s")
        
        log_file.write(final_message1 + '\n')
        log_file.write(f"Tempo total de treinamento: {total_training_time:.2f} segundos\n")
        log_file.write(f"Velocidade média: {avg_speed:.2f} episódios por segundo\n")
        
        # Resumo de performance final
        if recent_rewards:
            final_avg = np.mean(recent_rewards)
            final_successful = sum(1 for r in recent_rewards if r >= EARLY_STOP_THRESHOLD)
            final_success_rate = final_successful / len(recent_rewards)
            
            summary = f"""\n--- RESUMO DE PERFORMANCE FINAL ---
Episódios Executados: {episode + 1}
Velocidade Média: {avg_speed:.2f} ep/s
Recompensa Média Final: {final_avg:.2f}
Taxa de Sucesso Final: {final_success_rate:.2%}
Epsilon Final: {epsilon:.6f}
"""
            print(summary)
            log_file.write(summary + '\n')

        # Agrupa a matriz Q e os parâmetros para salvar em um único arquivo.
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
                'ALPHA': ALPHA,
                'GAMMA': GAMMA,
                'LAMBDA': LAMBDA,
                'MAX_STEPS': MAX_STEPS,
                'MASTER_SEED': MASTER_SEED
            }
        }
        
        # Salva o dicionário em um arquivo binário NumPy.
        np.save(OUTPUT_FILENAME, data_to_save)
        
        final_message2 = f"Matriz Q salva em '{OUTPUT_FILENAME}'."
        print(final_message2)
        log_file.write(final_message2 + '\n')

# Gerar gráfico da variação de recompensa usando o módulo grafico.py
print("\nGerando gráfico de desempenho...")
try:
    grafico.create_reward_graph(OUTPUT_FILENAME)
    print("Gráfico gerado com sucesso!")
except Exception as e:
    print(f"Erro ao gerar gráfico: {e}")
