import numpy as np
import math
import matplotlib.pyplot as plt

"""
@autor: Carlos Victor
@autor: Pedro Henrique Passos Rocha
@autor: Catterina Vittorazzi Salvador

OBS: O código utiliza a biblioteca matplotlib, numpy e math. Para instalar, utilize o comando: pip install matplotlib numpy
"""

# Função para calcular o coeficiente de determinação r^2
def calcular_r2(y_real, y_pred):
    ss_res = np.sum((y_real - y_pred) ** 2)
    ss_tot = np.sum((y_real - np.mean(y_real)) ** 2)
    return 1 - (ss_res / ss_tot)

def eliminacao_gauss(A, b):
    n = len(b)
    
    # Criar a matriz aumentada
    M = np.hstack([A, b.reshape(-1, 1)])
    
    # Eliminação de Gauss
    for i in range(n):
        # Pivotar a matriz
        for j in range(i + 1, n):
            fator = M[j, i] / M[i, i]
            M[j, i:] = M[j, i:] - fator * M[i, i:]
    
    # Substituição regressiva
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (M[i, -1] - np.dot(M[i, i+1:n], x[i+1:n])) / M[i, i]
    
    return x

# Função para ajustar o modelo linear: N = β0 + β1t
def ajuste_linear(t, N):
    A = np.vstack([t, np.ones(len(t))]).T
    b = N
    # Resolver o sistema linear A * [β1, β0] = N usando eliminação de Gauss
    coef = eliminacao_gauss(A.T @ A, A.T @ b)
    beta1, beta0 = coef
    N_pred = beta1 * t + beta0
    r2 = calcular_r2(N, N_pred)
    return N_pred, r2, beta1, beta0

# Função para ajustar o modelo quadrático: N = β0 + β1t + β2t^2
def ajuste_quadratico(t, N):
    A = np.vstack([t**2, t, np.ones(len(t))]).T
    b = N
    # Resolver o sistema linear A * [β2, β1, β0] = N usando eliminação de Gauss
    coef = eliminacao_gauss(A.T @ A, A.T @ b)
    beta2, beta1, beta0 = coef
    N_pred = beta2 * t**2 + beta1 * t + beta0
    r2 = calcular_r2(N, N_pred)
    return N_pred, r2, beta2, beta1, beta0

# Função para ajustar o modelo exponencial: N = β0 * exp(β1 * t)
def ajuste_exponencial(t, N):
    log_N = np.log(N)
    A = np.vstack([t, np.ones(len(t))]).T
    b = log_N
    # Resolver o sistema linear A * [β1, ln(β0)] = ln(N) usando eliminação de Gauss
    coef = eliminacao_gauss(A.T @ A, A.T @ b)
    beta1, log_beta0 = coef
    beta0 = math.exp(log_beta0)
    N_pred = beta0 * np.exp(beta1 * t)
    r2 = calcular_r2(N, N_pred)
    return N_pred, r2, beta0, beta1


# Dados fornecidos (idade em anos e quantidade de carbono-14)
t = np.array([77, 119, 205, 260, 343, 415, 425, 438, 502, 580, 604, 675, 
              696, 770, 802, 822, 897, 965, 970, 1027, 1094, 1156, 1192, 1282,
              1345, 1405, 1429, 1493, 1516, 1597, 1678, 1721, 1724, 1812, 1873, 
              1947, 1950, 2012, 2047, 2127, 2153, 2157, 2210, 2298, 2332, 2358, 2449, 2503])
N = np.array([50870643080, 46297918240, 38282822421, 34080460561, 28088573347, 24175635810, 23588757299, 22718540971, 19736321394, 
              16655820340, 15956176275, 13458062313, 13121778454, 10939070444, 10054385447, 9824068939, 8451625483, 7251508116, 
              7118718866, 6316609405, 5464716073, 4511225310, 4322167723, 3709676790, 3226060519, 2761872970, 2553745475, 2220343372, 
              2046714895, 1667362442, 1648561844, 1167013186, 1211599434, 1291077808, 1099297084, 791212548, 662664385, 721592837, 
              501015203, 536033559, 510953997, 583955494, 403003225, 574150933, 412508389, 131844628, 330342316, 326167250])

def main():
    
    print("\tTrabalho de algoritmos")
    # Ajuste Linear
    N_pred_linear, r2_linear, beta1_linear, beta0_linear = ajuste_linear(t, N)
    print(f"\tAjuste Linear: r² = {r2_linear}, β1 = {beta1_linear}, β0 = {beta0_linear}")

    # Ajuste Quadrático
    N_pred_quad, r2_quad, beta2_quad, beta1_quad, beta0_quad = ajuste_quadratico(t, N)
    print(f"\tAjuste Quadrático: r² = {r2_quad}, β2 = {beta2_quad}, β1 = {beta1_quad}, β0 = {beta0_quad}")

    # Ajuste Exponencial
    N_pred_exp, r2_exp, beta0_exp, beta1_exp = ajuste_exponencial(t, N)
    print(f"\tAjuste Exponencial: r² = {r2_exp}, β0 = {beta0_exp}, β1 = {beta1_exp}")

    melhor_r2 = max(r2_linear, r2_quad, r2_exp)
    
    if melhor_r2 == r2_linear:
        melhor_ajuste = "linear"
    elif melhor_r2 == r2_quad:
        melhor_ajuste = "quadrático"
    else:
        melhor_ajuste = "exponencial"
    
    print(f"\nO melhor ajuste foi o {melhor_ajuste} com r² = {melhor_r2:.4f}")

    plt.figure(figsize=(10, 6))
    plt.scatter(t, N, label='Dados', color='black')
    plt.plot(t, N_pred_linear, label=f'Ajuste Linear (r² = {r2_linear:.4f})', color='green')
    plt.plot(t, N_pred_quad, label=f'Ajuste Quadrático (r² = {r2_quad:.4f})', color='orange')
    plt.plot(t, N_pred_exp, label=f'Ajuste Exponencial (r² = {r2_exp:.4f})', color='red')
    plt.xlabel('Idade (t)')
    plt.ylabel('Quantidade de Carbono-14 (N)')
    plt.title('Ajuste de Curvas para Carbono-14')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()