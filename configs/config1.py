# Parâmetros para treino do Q(λ) - Experimento 1

# --- Hiperparâmetros do Algoritmo Q(λ) ---
ALPHA = 0.09 # Taxa de aprendizado (learning rate)
GAMMA = 0.97 # Fator de desconto para recompensas futuras
LAMBDA = 0.8 # Fator de decaimento para os rastros de elegibilidade
# --- Parâmetros de Exploração (Epsilon-Greedy) ---
EPSILON = 1.0
EPSILON_DECAY_RATE = 5e-5
MIN_EPSILON = 0.0001
# --- Parâmetros de Saída ---
FILENAME_BASE = "treino_Qlambda"

# --- CONTROLE DE SEEDS PARA REPRODUTIBILIDADE ---
MASTER_SEED = 17  # Seed principal para reprodutibilidade

# --- Parâmetros Físicos Configuráveis ---
GRAVITY = 10.0
FORCE_MAGNITUDE = 1.5

# --- Restrições Físicas Impostas ao Problema ---
POSITION_LIMIT = 1
ANGLE_LIMIT_RADS = 0.5
VELOCITY_LIMIT = 3
ANGULAR_VELOCITY_LIMIT = 3

# --- Parâmetros da Discretização do Espaço de Estados ---
N_POSITION = 5
N_VELOCITY = 5
N_ANGLE = 7
N_ANGULAR_VELOCITY = 7

# --- Parâmetros de Treinamento ---
NUM_EPISODES = 4000
MAX_STEPS = 1000 # Número máximo de passos por episódio

# --- Parâmetros de Early Stopping ---
EARLY_STOP_THRESHOLD = 900
EARLY_STOP_WINDOW = 300
EARLY_STOP_SUCCESS_RATE = 0.98
MIN_EPISODES = 200
PLATEAU_WINDOW = 500
PLATEAU_TOLERANCE = 10