Grupo:
- Pedro Ulisses de Lima Quadros
- Iago Jacob de Souza Ramos
- Leonardo Fernandes Trevilato
- Rhuan dos Santos Vicente

O pêndulo invertido é um exercício famoso para o estudo de aprendizado por reforço. O jogo consiste em tentar equilibrar um pêndulo invertido em cima de um carrinho apenas exercendo duas forças horizontais, por meio dos botões 'z' e 'x'. Para jogar, apenas execute "$python3 main.py" em seu terminal. Utilizamos o algoritmo TD-Lambda causal para estimar os valores esperados da função valor em cada estado para a política nmanual (i.e. a estratégia utilizada pelo jogador).

***cost_values.txt:***  vetor das estimativas do valor esperado do ganho atualizado por Robins-Monro a cada vez que o jogo é jogado.

***custom_termination_wrapper.py:***    Wrapper para a customização das condições de falha do pêndulo, e.g. angulo máximo, distância da parede ao centro, etc.

***td-lambda.py:*** Classe que roda TD-lambda conforme controlamos o pêndulo. É instanciada em main.py.

***main.py:***  Rotina principal em que o jogo é jogado. Enquanto jogamos, os valores esperados de V são estimados por TD-lambda e, ao fim, são guardados em *cost_values.txt*.

    ***Quão difícil é controlar esse sistema?***
        Muito! Trata-se de um sistema caótico. Tente por si só e você verá ;)

    ***Como o progresso no aprendizado do controle manual poderia ser estimado?***
        Registrando-se a função de valor conforme mais partidas são jogadas. Uma alta função de valor indica um melhor desempenho.
