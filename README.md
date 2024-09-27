# Ajuste de Curvas para Carbono-14

Este código foi desenvolvido para ajustar modelos matemáticos (linear, quadrático e exponencial) aos dados de decaimento de carbono-14. Ele utiliza as bibliotecas `numpy`, `math` e `matplotlib` para realizar cálculos e exibir gráficos.

## Funcionalidades:

- **Eliminação de Gauss**: Implementa o método de eliminação de Gauss para resolver sistemas lineares.
- **Ajustes**: Três tipos de ajuste são realizados:
  - Ajuste Linear: \( N = \beta_0 + \beta_1 t \)
  - Ajuste Quadrático: \( N = \beta_0 + \beta_1 t + \beta_2 t^2 \)
  - Ajuste Exponencial: \( N = \beta_0 \cdot e^{\beta_1 t} \)
- **Coeficiente de Determinação \( r^2 \)**: Calcula a qualidade dos ajustes por meio do coeficiente \( r^2 \).

## Execução

1. O código ajusta os três modelos aos dados fornecidos.
2. Compara os valores de \( r^2 \) para escolher o melhor modelo.
3. Gera um gráfico que mostra os dados e os ajustes.

## Requisitos

Instale as bibliotecas necessárias com o comando:

```bash
pip install numpy matplotlib
