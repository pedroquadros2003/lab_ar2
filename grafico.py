import numpy as np
import matplotlib
matplotlib.use('Agg')  # Usa backend sem interface gráfica
import matplotlib.pyplot as plt
import os
import argparse

def read_training_log(filename):
    """
    Lê o arquivo de log de treinamento e extrai as taxas de sucesso por episódio.
    """
    success_rates = []
    episodes = []
    
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                # Procura por linhas que contêm "Episódio" e "Taxa Sucesso" (formato correto do log)
                if "Episódio:" in line and "Taxa Sucesso:" in line:
                    try:
                        # Extrai o número do episódio
                        episode_part = line.split("Episódio:")[1].split(",")[0].strip()
                        episode_num = int(episode_part.split("/")[0])
                        
                        # Extrai a taxa de sucesso
                        success_part = line.split("Taxa Sucesso:")[1].split("%")[0].strip()
                        success_rate = float(success_part)
                        
                        episodes.append(episode_num)
                        success_rates.append(success_rate)
                    except (ValueError, IndexError):
                        # Pula linhas mal formatadas
                        continue
    except FileNotFoundError:
        print(f"Arquivo não encontrado: {filename}")
        return [], []
    except Exception as e:
        print(f"Erro ao ler arquivo {filename}: {e}")
        return [], []
    
    return episodes, success_rates

def find_file_in_folders(filename, search_folders=['npy', 'trainLog', '.']):
    """
    Procura um arquivo em várias pastas.
    """
    for folder in search_folders:
        file_path = os.path.join(folder, filename)
        if os.path.exists(file_path):
            return file_path
    return None

def create_reward_graph(filename, output_name=None):
    """
    Cria um gráfico de taxa de sucesso por episódio a partir de um arquivo de log.
    """
    # Cria a pasta graficos se não existir
    graficos_dir = "graficos"
    if not os.path.exists(graficos_dir):
        os.makedirs(graficos_dir)
        print(f"Pasta '{graficos_dir}' criada.")
    
    # Procura o arquivo .npy nas pastas
    npy_file_path = find_file_in_folders(filename, ['npy', '.'])
    if not npy_file_path:
        print(f"Arquivo .npy não encontrado: {filename}")
        print("Procurado em: npy/, diretório atual")
        return
    
    # Determina o nome do arquivo de log
    base_name = os.path.splitext(os.path.basename(filename))[0]
    log_filename = f"{base_name}.txt"
    
    # Procura o arquivo de log nas pastas
    log_file_path = find_file_in_folders(log_filename, ['trainLog', '.'])
    if not log_file_path:
        print(f"Arquivo de log não encontrado: {log_filename}")
        print("Procurado em: trainLog/, diretório atual")
        return
    
    # Lê os dados do log
    episodes, success_rates = read_training_log(log_file_path)
    
    if not episodes:
        print(f"Nenhum dado de taxa de sucesso encontrado em {log_file_path}")
        return
    
    # Carrega o arquivo .npy para obter os parâmetros
    try:
        data = np.load(npy_file_path, allow_pickle=True).item()
        params = data.get('params', {})
        lambda_val = params.get('LAMBDA', 'N/A')
        alpha = params.get('ALPHA', 'N/A')
        gamma = params.get('GAMMA', 'N/A')
    except Exception as e:
        print(f"Erro ao carregar parâmetros de {filename}: {e}")
        lambda_val = alpha = gamma = 'N/A'
    
    # Cria o gráfico
    plt.figure(figsize=(12, 8))
    
    # Gráfico principal - linha com todos os pontos
    plt.plot(episodes, success_rates, 'b-', linewidth=1, alpha=0.7, label='Taxa de sucesso por episódio')
    
    # Adiciona média móvel se houver dados suficientes
    if len(success_rates) > 100:
        # Calcula média móvel de 100 episódios
        window_size = min(100, len(success_rates) // 10)
        moving_avg = []
        for i in range(len(success_rates)):
            start_idx = max(0, i - window_size + 1)
            avg = np.mean(success_rates[start_idx:i+1])
            moving_avg.append(avg)
        
        plt.plot(episodes, moving_avg, 'r-', linewidth=2, 
                label=f'Média móvel ({window_size} episódios)')
    
    # Adiciona linhas de referência
    plt.axhline(y=50, color='orange', linestyle='--', alpha=0.7, label='50% de sucesso')
    plt.axhline(y=90, color='green', linestyle='--', alpha=0.7, label='90% de sucesso')
    
    # Configurações do gráfico
    plt.xlabel('Episódios', fontsize=12)
    plt.ylabel('Taxa de Sucesso (%)', fontsize=12)
    plt.ylim(0, 100)
    
    # Título com informações dos parâmetros
    if lambda_val != 'N/A':
        title = f'Evolução da Taxa de Sucesso - Q(λ) no Pêndulo Invertido\n'
        title += f'λ={lambda_val}, α={alpha}, γ={gamma}'
    else:
        title = f'Evolução da Taxa de Sucesso - {base_name}'
    
    plt.title(title, fontsize=14, pad=20)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Adiciona estatísticas no gráfico
    max_success = max(success_rates)
    avg_success = np.mean(success_rates)
    final_avg = np.mean(success_rates[-min(100, len(success_rates)):])  # Média dos últimos 100 episódios
    
    stats_text = f'Max: {max_success:.1f}%\nMédia Geral: {avg_success:.1f}%\nMédia Final: {final_avg:.1f}%'
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Define nome do arquivo de saída
    if output_name is None:
        output_name = os.path.join(graficos_dir, f"{base_name}_sucesso.png")
    else:
        # Se foi especificado um nome personalizado, também salva na pasta graficos
        if not os.path.dirname(output_name):
            output_name = os.path.join(graficos_dir, output_name)
    
    # Salva o gráfico
    plt.savefig(output_name, dpi=300, bbox_inches='tight')
    # plt.show()  # Comentado para evitar problemas com Tkinter
    
    print(f"Gráfico salvo em: {output_name}")
    print(f"Estatísticas:")
    print(f"  - Total de episódios: {len(episodes)}")
    print(f"  - Taxa de sucesso máxima: {max_success:.2f}%")
    print(f"  - Taxa de sucesso média: {avg_success:.2f}%")
    print(f"  - Taxa de sucesso média (últimos 100 ep.): {final_avg:.2f}%")

def list_available_files():
    """
    Lista todos os arquivos .npy disponíveis em várias pastas.
    """
    search_folders = ['npy', '.']
    npy_files = []
    
    for folder in search_folders:
        if os.path.exists(folder):
            folder_files = [f for f in os.listdir(folder) if f.endswith('.npy') and f.startswith('treino_')]
            # Adiciona o caminho da pasta aos nomes dos arquivos para identificação
            for f in folder_files:
                display_name = f if folder == '.' else f
                full_path = f if folder == '.' else os.path.join(folder, f)
                npy_files.append((display_name, full_path, folder))
    
    if npy_files:
        print("Arquivos de treinamento disponíveis:")
        for i, (display_name, full_path, folder) in enumerate(npy_files, 1):
            # Verifica se existe o arquivo de log correspondente
            base_name = os.path.splitext(display_name)[0]
            log_filename = f"{base_name}.txt"
            log_path = find_file_in_folders(log_filename, ['trainLog', '.'])
            status = "✓" if log_path else "✗ (sem log)"
            location = f"({folder}/" if folder != '.' else "(raiz/"
            print(f"  {i}. {display_name} {location}) {status}")
    else:
        print("Nenhum arquivo de treinamento (.npy) encontrado.")
        print("Procurado em: npy/, diretório atual")
    
    return [item[1] for item in npy_files]  # Retorna apenas os caminhos completos

def main():
    parser = argparse.ArgumentParser(description='Gera gráfico de taxa de sucesso por episódio a partir de arquivos de treinamento')
    parser.add_argument('arquivo', nargs='?', help='Nome do arquivo .npy de treinamento')
    parser.add_argument('-o', '--output', help='Nome do arquivo de saída (PNG)')
    parser.add_argument('-l', '--listar', action='store_true', help='Lista arquivos disponíveis')
    
    args = parser.parse_args()
    
    if args.listar:
        list_available_files()
        return
    
    # Se nenhum arquivo foi especificado, lista os disponíveis e pede para escolher
    if not args.arquivo:
        available_files = list_available_files()
        if not available_files:
            return
        
        try:
            print("\nEscolha um arquivo:")
            choice = int(input("Digite o número do arquivo: ")) - 1
            if 0 <= choice < len(available_files):
                args.arquivo = available_files[choice]
            else:
                print("Escolha inválida.")
                return
        except (ValueError, KeyboardInterrupt):
            print("Operação cancelada.")
            return
    
    # Verifica se o arquivo existe
    file_path = find_file_in_folders(args.arquivo, ['npy', '.'])
    if not file_path:
        print(f"Arquivo não encontrado: {args.arquivo}")
        print("Use --listar para ver os arquivos disponíveis.")
        return
    
    # Gera o gráfico
    create_reward_graph(args.arquivo, args.output)

if __name__ == "__main__":
    main()