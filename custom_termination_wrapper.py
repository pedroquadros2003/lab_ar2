import gymnasium as gym

# Wrapper para modificar a condição de término do ambiente
class CustomTerminationWrapper(gym.Wrapper):
    def __init__(self, env, angle_limit, pos_limit, vel_limit, ang_vel_limit):
        super().__init__(env)
        self.angle_limit = angle_limit
        self.pos_limit = pos_limit
        self.vel_limit = vel_limit
        self.ang_vel_limit = ang_vel_limit

    def step(self, action):
        # Executa o passo no ambiente original
        observation, reward, terminated, truncated, info = self.env.step(action)

        # Ignora a condição de término original e cria a nossa com base nos limites.
        # observation = [posição, ângulo, velocidade, vel. angular]
        pos, angle, vel, ang_vel = observation

        out_of_bounds = (
            abs(pos) > self.pos_limit or
            abs(angle) > self.angle_limit or
            abs(vel) > self.vel_limit or
            abs(ang_vel) > self.ang_vel_limit
        )
        
        terminated = out_of_bounds

        return observation, reward, terminated, truncated, info
