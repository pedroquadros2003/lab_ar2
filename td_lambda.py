import numpy as np

class TDLambdaCausal():
   
    def __init__(self, dims, alpha, lambda_, gamma, 
                 pos_limit, angle_limit, vel_limit, ang_vel_limit):
        
        self.alpha = alpha                              # Taxa de aprendizado
        self.lambda_ = lambda_                          # Fator de decaimento do rastro (usamos lambda_ para não conflitar com a palavra-chave do Python)
        self.gamma = gamma                              # Fator de desconto
        self.dim = dims                                 # Dimensão da discretização (ex: (10, 10, 10, 10))

        # Armazena as restrições físicas como atributos da classe
        self.pos_limit = pos_limit
        self.angle_limit = angle_limit
        self.vel_limit = vel_limit
        self.ang_vel_limit = ang_vel_limit

        # Cria a matriz de custo com valores aleatórios no interior e zeros nas bordas
        self.cost_matrix = np.random.uniform(-10, 10, size=self.dim)
        self.cost_matrix[0, :, :, :] = 0
        self.cost_matrix[-1, :, :, :] = 0
        self.cost_matrix[:, 0, :, :] = 0
        self.cost_matrix[:, -1, :, :] = 0
        self.cost_matrix[:, :, 0, :] = 0
        self.cost_matrix[:, :, -1, :] = 0
        self.cost_matrix[:, :, :, 0] = 0
        self.cost_matrix[:, :, :, -1] = 0


        # Rastros de elegibilidade, um para cada estado
        self.eligibility_traces = np.zeros(self.dim)

    
    def state_cost(self, state):
        return self.cost_matrix[state[0], state[1], state[2], state[3]]
    


    def convert2state(self, v):
        
        i0_raw = ( v[0] + self.pos_limit ) / (2*self.pos_limit) * (self.dim[0] - 1)
        i1_raw = ( v[1] + self.angle_limit ) / (2*self.angle_limit) * (self.dim[1] - 1)
        i2_raw = ( v[2] + self.vel_limit ) / (2*self.vel_limit) * (self.dim[2] - 1)
        i3_raw = ( v[3] + self.ang_vel_limit ) / (2*self.ang_vel_limit) * (self.dim[3] - 1)


        i0 = int(np.clip(i0_raw, 0, self.dim[0] - 1))
        i1 = int(np.clip(i1_raw, 0, self.dim[1] - 1))
        i2 = int(np.clip(i2_raw, 0, self.dim[2] - 1))
        i3 = int(np.clip(i3_raw, 0, self.dim[3] - 1))

        return [i0, i1, i2, i3]

    def update(self, prev_state_idx, current_state_idx, reward):
        # 1. Pega o valor do estado anterior e do estado atual
        v_prev = self.state_cost(prev_state_idx)
        v_current = self.state_cost(current_state_idx)

        # 2. Calcula o erro de diferença temporal (TD Error)
        delta = reward + self.gamma * v_current - v_prev

        # 3. Incrementa o rastro do estado anterior
        self.eligibility_traces[prev_state_idx[0], prev_state_idx[1], prev_state_idx[2], prev_state_idx[3]] += 1

        # 4. Atualiza a função de valor e decai os rastros para todos os estados
        self.cost_matrix += self.alpha * delta * self.eligibility_traces
        self.eligibility_traces *= self.gamma * self.lambda_
