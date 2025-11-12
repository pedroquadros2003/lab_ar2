import numpy as np

class QLambdaCausal():
    """
    Implementa o algoritmo Q(λ) online.
    Este algoritmo aprende uma função de valor de ação Q(s, a) usando rastros de elegibilidade.
    """
    
    def __init__(self, dims, num_actions, alpha, lambda_, gamma, 
                 pos_limit, angle_limit, vel_limit, ang_vel_limit):
        
        self.alpha = alpha                                  # Taxa de aprendizado
        self.lambda_ = lambda_                              # Fator de decaimento do rastro
        self.gamma = gamma                                  # Fator de desconto
        
        # Dimensões do espaço de estados e número de ações
        self.state_dims = dims
        self.num_actions = num_actions
        self.q_dims = self.state_dims + (self.num_actions,) # Adiciona a dimensão da ação

        # Armazena as restrições físicas
        self.pos_limit = pos_limit
        self.angle_limit = angle_limit
        self.vel_limit = vel_limit
        self.ang_vel_limit = ang_vel_limit

        # Matriz Q(s, a) com valores aleatórios
        self.q_matrix = np.random.uniform(-1, 1, size=self.q_dims)
        
        # Rastros de elegibilidade para cada par (estado, ação)
        self.eligibility_traces = np.zeros(self.q_dims)

    def get_q_value(self, state_idx, action_idx):
        """Retorna o valor Q para um estado e ação específicos."""
        # state_idx deve ser uma tupla para indexação avançada
        return self.q_matrix[state_idx][action_idx]

    def get_v_value(self, state_idx):
        """Calcula o valor de um estado (V(s)) como o máximo dos valores Q para aquele estado."""
        # state_idx deve ser uma tupla
        return np.max(self.q_matrix[state_idx])

    def convert2state(self, observation):
        """Discretiza a observação contínua em um índice de estado (tupla)."""
        i0_raw = (observation[0] + self.pos_limit) / (2 * self.pos_limit) * (self.state_dims[0] - 1)
        i1_raw = (observation[1] + self.angle_limit) / (2 * self.angle_limit) * (self.state_dims[1] - 1)
        i2_raw = (observation[2] + self.vel_limit) / (2 * self.vel_limit) * (self.state_dims[2] - 1)
        i3_raw = (observation[3] + self.ang_vel_limit) / (2 * self.ang_vel_limit) * (self.state_dims[3] - 1)

        i0 = int(np.clip(i0_raw, 0, self.state_dims[0] - 1))
        i1 = int(np.clip(i1_raw, 0, self.state_dims[1] - 1))
        i2 = int(np.clip(i2_raw, 0, self.state_dims[2] - 1))
        i3 = int(np.clip(i3_raw, 0, self.state_dims[3] - 1))

        return (i0, i1, i2, i3)

    def choose_action(self, state_idx, epsilon):
        """
        Escolhe uma ação usando a estratégia epsilon-greedy.

        Args:
            state_idx (tuple): O índice do estado atual.
            epsilon (float): A probabilidade de escolher uma ação aleatória (exploração).

        Returns:
            int: O índice da ação escolhida.
        """
        # Com probabilidade epsilon, escolhe uma ação aleatória (exploração)
        if np.random.random() < epsilon:
            return np.random.randint(0, self.num_actions)
        # Caso contrário, escolhe a melhor ação conhecida (explotação)
        else:
            state_idx = tuple(state_idx)
            # Retorna o índice da ação com o maior valor Q para o estado atual
            return np.argmax(self.q_matrix[state_idx])

    def update(self, prev_state_idx, prev_action_idx, reward, current_state_idx):
        """
        Atualiza a matriz Q usando o algoritmo Q(λ) de Peng e Williams, mas
        com a simplificação de não utilizar a lista H de pares estado-ação
        utilizados.
        
        Args:
            prev_state_idx (tuple): Índice do estado anterior (s_t).
            prev_action_idx (int): Índice da ação executada (a_t).
            reward (float): Recompensa recebida (r_{t+1}).
            current_state_idx (tuple): Índice do estado atual (s_{t+1}).
        """
        # Garante que os índices de estado sejam tuplas
        prev_state_idx = tuple(prev_state_idx)
        current_state_idx = tuple(current_state_idx)

        # --- Passos do Algoritmo Q(λ) ---

        # Calculam-se os valores de V(s_t), V(s_t+1) e Q(s_t, a_t).
        v_prev = self.get_v_value(prev_state_idx)
        v_current = self.get_v_value(current_state_idx)
        q_prev = self.get_q_value(prev_state_idx, prev_action_idx)

        # e'_t: Erro para o par (s_t, a_t) específico (Passo 1)
        e_prime_t = reward + self.gamma * v_current - q_prev
        # e_t: Erro para o estado s_t, usado para os outros pares (Passo 2)
        e_t = reward + self.gamma * v_current - v_prev

        # Atualiza-se Q para todos os pares (s,a) usando o erro e_t e decai os rastros
        
        # e(s,a) <- γλ * e(s,a)
        self.eligibility_traces *= self.gamma * self.lambda_  # (Passo 3a) 

        # Q(s,a) <- Q(s,a) + α * e_t * e(s,a)
        self.q_matrix += self.alpha * e_t * self.eligibility_traces # (Passo 3b)
        
        # Atualiza Q(s_t, a_t) com o erro específico e'_t
        # Q(s_t, a_t) <- Q(s_t, a_t) + α * e'_t
        q_update_idx = prev_state_idx + (prev_action_idx,)
        self.q_matrix[q_update_idx] += self.alpha * e_prime_t # (Passo 4)

        # Incrementa o rastro de elegibilidade para o par (s_t, a_t) visitado
        # e(s_t, a_t) <- e(s_t, a_t) + 1
        trace_idx = prev_state_idx + (prev_action_idx,)
        self.eligibility_traces[trace_idx] += 1 # (Passo 5)
