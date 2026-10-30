# -*- coding: utf-8 -*-
"""
Análise de estabilidade global de estruturas pré-moldadas — Hogeslag e gamma_z

Modelo implementado a partir do enunciado fornecido pelo usuário.

HIPÓTESES DO MODELO:
- Pórtico plano com quatro pilares contínuos e três vãos de 18 m por nível.
- Ligações viga-pilar articuladas: não há transmissão de rigidez à rotação pelas vigas.
- Vigas axialmente indeformáveis: a translação horizontal é comum aos quatro pilares em cada nível.
- As rotações dos quatro pilares são graus de liberdade independentes em cada nível.
- Pilares modelados por elementos de Euler-Bernoulli, com pé-direito de 6 m.
- Bases F1: translação e rotação impedidas.
- Bases F2: translação impedida e uma mola rotacional de 75 MN.m/rad em cada pilar.
- Cargas permanentes e variáveis das vigas convertidas em reações de apoio por vão.
- Excentricidade das reações representada por momentos nodais com sinais opostos nos apoios
  esquerdo e direito de cada vão.
- Peso próprio dos pilares tratado como carga vertical distribuída ao longo dos elementos.

LIMITAÇÕES DELIBERADAS:
- O programa calcula a análise linear de primeira ordem e os indicadores aproximados.
- Não substitui uma análise normativa completa nem uma análise não linear P-Delta.
- O comprimento de flambagem l_e e a altura H_eq de Hogeslag são entradas explícitas,
  pois o enunciado não fornece uma equivalência validada para cargas em vários níveis.
- A imperfeição global é opcional e informada pelo usuário em milirradianos. O valor deve
  ser definido com base no critério normativo adotado; o programa não escolhe esse valor.
- Os fatores de rigidez são entradas explícitas. Não se presume automaticamente EI/3.

Execute com: streamlit run tarefa_5_hogeslag_corrigido.py
"""

from __future__ import annotations

from itertools import product
from math import pi
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st


# =============================================================================
# 1. DADOS DO ENUNCIADO
# =============================================================================

N_COLUNAS = 4
N_VAOS = 3
L_VAO = 18.0                 # m
H_PAVIMENTO = 6.0            # m
B_PILAR = 0.40               # m
H_PILAR = 0.40               # m
EC = 30.0e6                  # kN/m² (30 GPa)
GAMMA_CONCRETO = 25.0        # kN/m³ — não usado para recalcular gk das vigas
G_K_VIGA = 13.0              # kN/m
Q_K_VIGA = 7.0               # kN/m
PESO_PROPRIO_PILAR_KN_M = 4.0  # kN/m por pilar
EXCENTRICIDADE = 0.40        # m
K_ROTACIONAL_PILAR = 75_000.0  # kN.m/rad por pilar (75 MN.m/rad)

I_PILAR = B_PILAR * H_PILAR**3 / 12.0

COMBINACOES = {
    # gamma_G, gamma_Q efetivo, gamma_W efetivo
    "C1": {"gamma_g": 1.4, "gamma_q": 1.4, "gamma_w": 0.0,
           "descricao": "Q principal, sem vento"},
    "C2": {"gamma_g": 1.4, "gamma_q": 1.4, "gamma_w": 0.84,
           "descricao": "Q principal, vento acompanhante (psi0,W = 0,6)"},
    "C3": {"gamma_g": 1.4, "gamma_q": 1.12, "gamma_w": 1.4,
           "descricao": "Vento principal, Q acompanhante (psi0,Q = 0,8)"},
    "C4": {"gamma_g": 1.4, "gamma_q": 0.0, "gamma_w": 1.4,
           "descricao": "Vento principal, sem Q"},
}


# =============================================================================
# 2. FUNÇÕES DE ELEMENTO E GRAUS DE LIBERDADE
# =============================================================================

def matriz_rigidez_flexao(EI: float, comprimento: float) -> np.ndarray:
    """Matriz local de rigidez Euler-Bernoulli [v_i, theta_i, v_j, theta_j]."""
    L = comprimento
    return (EI / L**3) * np.array(
        [
            [12.0, 6.0 * L, -12.0, 6.0 * L],
            [6.0 * L, 4.0 * L**2, -6.0 * L, 2.0 * L**2],
            [-12.0, -6.0 * L, 12.0, -6.0 * L],
            [6.0 * L, 2.0 * L**2, -6.0 * L, 4.0 * L**2],
        ],
        dtype=float,
    )

def matriz_rigidez_geometrica(N_compressao: float, comprimento: float) -> np.ndarray:
    """Matriz local de rigidez geométrica para compressão [v_i, theta_i, v_j, theta_j]."""
    L = comprimento
    # Utilizando a matriz consistente de energia de deformação geométrica
    return (N_compressao / (30.0 * L)) * np.array([
        [ 36.0,      3.0 * L,     -36.0,      3.0 * L],
        [ 3.0 * L,   4.0 * L**2,  -3.0 * L,  -L**2],
        [-36.0,     -3.0 * L,      36.0,     -3.0 * L],
        [ 3.0 * L,  -L**2,        -3.0 * L,   4.0 * L**2]
    ], dtype=float)

def vetor_carga_distribuida_transversal(q: float, comprimento: float) -> np.ndarray:
    """Vetor nodal consistente para carga transversal uniforme q no sentido +v."""
    L = comprimento
    return q * np.array([L / 2.0, L**2 / 12.0, L / 2.0, -L**2 / 12.0])


def indices_globais(coluna: int, nivel_inferior: int, n_pav: int) -> list[int]:
    """DOFs locais [u_i, theta_i, u_j, theta_j]. u é comum aos pilares por nível."""
    n_nos = n_pav + 1
    idx_u_i = nivel_inferior
    idx_t_i = n_nos + coluna * n_nos + nivel_inferior
    idx_u_j = nivel_inferior + 1
    idx_t_j = n_nos + coluna * n_nos + nivel_inferior + 1
    return [idx_u_i, idx_t_i, idx_u_j, idx_t_j]


def idx_rotacao(coluna: int, nivel: int, n_pav: int) -> int:
    n_nos = n_pav + 1
    return n_nos + coluna * n_nos + nivel


def numero_graus_liberdade(n_pav: int) -> int:
    return (n_pav + 1) + N_COLUNAS * (n_pav + 1)


# =============================================================================
# 3. AÇÕES VERTICAIS, REAÇÕES E MOMENTOS EXCÊNTRICOS
# =============================================================================

def calcular_acoes_verticais(
    n_pav: int,
    mascara_q: np.ndarray,
    combinacao: str,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """
    Calcula, por nível e pilar:
    - reações verticais de vigas;
    - momentos nodais decorrentes da excentricidade.

    Convenção de sinais dos momentos excêntricos:
    - reação do vão à direita do pilar: -e.R;
    - reação do vão à esquerda do pilar: +e.R.

    Assim, reações iguais dos dois vãos adjacentes se anulam no pilar interno.
    """
    fatores = COMBINACOES[combinacao]
    reacoes = np.zeros((n_pav + 1, N_COLUNAS), dtype=float)
    momentos = np.zeros((n_pav + 1, N_COLUNAS), dtype=float)

    g_d = fatores["gamma_g"] * G_K_VIGA
    q_d_fator = fatores["gamma_q"] * Q_K_VIGA

    for nivel in range(1, n_pav + 1):
        for vao in range(N_VAOS):
            q_d = g_d + q_d_fator * float(mascara_q[nivel - 1, vao])
            reacao = q_d * L_VAO / 2.0

            pilar_esquerdo = vao
            pilar_direito = vao + 1

            reacoes[nivel, pilar_esquerdo] += reacao
            reacoes[nivel, pilar_direito] += reacao

            # Cada vão gera um par de momentos opostos nos seus apoios.
            momentos[nivel, pilar_esquerdo] -= EXCENTRICIDADE * reacao
            momentos[nivel, pilar_direito] += EXCENTRICIDADE * reacao

    peso_pilar_d = fatores["gamma_g"] * PESO_PROPRIO_PILAR_KN_M
    peso_total_pilares = N_COLUNAS * n_pav * H_PAVIMENTO * peso_pilar_d
    carga_viga_total = float(np.sum(reacoes))
    carga_vertical_total = carga_viga_total + peso_total_pilares

    return reacoes, momentos, peso_pilar_d, carga_vertical_total


def distribuicao_vento(n_pav: int, tipo_vento: str) -> np.ndarray:
    """Forças características por nível, conforme V1 ou V2 do enunciado."""
    cotas = H_PAVIMENTO * np.arange(1, n_pav + 1, dtype=float)
    if tipo_vento == "V1":
        return np.full(n_pav, 80.0, dtype=float)
    if tipo_vento == "V2":
        return 80.0 * cotas / np.sum(cotas)
    raise ValueError(f"Tipo de vento desconhecido: {tipo_vento}")


# =============================================================================
# 4. MONTAGEM E SOLUÇÃO DO MODELO ESTRUTURAL
# =============================================================================

def montar_e_resolver_modelo(
    n_pav: int,
    fundacao: str,
    reacoes_vigas: np.ndarray,
    momentos_exc: np.ndarray,
    peso_pilar_d: float,
    vento_caracteristico: np.ndarray,
    fator_vento: float,
    sinal: int,
    theta_imperfeicao_rad: float,
    fator_rigidez_primeira_ordem: float,
    max_iter: int = 15,
    tolerancia: float = 1e-5
) -> dict[str, Any]:
    
    ndof = numero_graus_liberdade(n_pav)
    K_elastica = np.zeros((ndof, ndof), dtype=float)
    F = np.zeros(ndof, dtype=float)

    EI_coluna = EC * I_PILAR * fator_rigidez_primeira_ordem
    k_elastica_local = matriz_rigidez_flexao(EI_coluna, H_PAVIMENTO)

    dofs_elementos: dict[tuple[int, int], list[int]] = {}
    cargas_equiv_elementos: dict[tuple[int, int], np.ndarray] = {}
    forcas_normais_estaticas: dict[tuple[int, int], float] = {}

    # 1. Montagem da Matriz Elástica, Forças Normais e Cargas Nocionais
    for coluna in range(N_COLUNAS):
        for trecho in range(n_pav):
            dofs = indices_globais(coluna, trecho, n_pav)
            dofs_elementos[(coluna, trecho)] = dofs
            K_elastica[np.ix_(dofs, dofs)] += k_elastica_local

            # Força normal estática média no trecho (para a matriz geométrica)
            nivel_superior = trecho + 1
            cargas_vigas_acima = float(np.sum(reacoes_vigas[nivel_superior:, coluna]))
            peso_trechos_acima = peso_pilar_d * H_PAVIMENTO * (n_pav - nivel_superior)
            N_topo = cargas_vigas_acima + peso_trechos_acima
            N_base = N_topo + peso_pilar_d * H_PAVIMENTO
            forcas_normais_estaticas[(coluna, trecho)] = (N_topo + N_base) / 2.0

            q_h_imp = peso_pilar_d * theta_imperfeicao_rad * sinal
            f_eq = vetor_carga_distribuida_transversal(q_h_imp, H_PAVIMENTO)
            cargas_equiv_elementos[(coluna, trecho)] = f_eq
            F[dofs] += f_eq

    # Condições de base F2
    if fundacao == "F2":
        for coluna in range(N_COLUNAS):
            dof_theta_base = idx_rotacao(coluna, 0, n_pav)
            K_elastica[dof_theta_base, dof_theta_base] += K_ROTACIONAL_PILAR
    elif fundacao != "F1":
        raise ValueError("Fundação deve ser F1 ou F2.")

    # Ações Externas Nodais
    for nivel in range(1, n_pav + 1):
        for coluna in range(N_COLUNAS):
            F[idx_rotacao(coluna, nivel, n_pav)] += momentos_exc[nivel, coluna]
            
    vento_d = fator_vento * vento_caracteristico * sinal
    for nivel in range(1, n_pav + 1):
        F[nivel] += vento_d[nivel - 1]
        p_nivel = float(np.sum(reacoes_vigas[nivel, :]))
        F[nivel] += sinal * theta_imperfeicao_rad * p_nivel

    # Fixação dos graus de liberdade restritos
    fixos = [0] 
    if fundacao == "F1":
        fixos.extend(idx_rotacao(c, 0, n_pav) for c in range(N_COLUNAS))
    livres = np.array([i for i in range(ndof) if i not in set(fixos)], dtype=int)
    
    F_l = F[livres]
    desloc = np.zeros(ndof, dtype=float)

    # 2. Laço Iterativo P-Delta
    condicao_matriz = 0.0
    for iteracao in range(max_iter):
        # Reconstrói a Matriz Geométrica a cada passo
        K_geometrica = np.zeros((ndof, ndof), dtype=float)
        
        for coluna in range(N_COLUNAS):
            for trecho in range(n_pav):
                dofs = dofs_elementos[(coluna, trecho)]
                N_comp = forcas_normais_estaticas[(coluna, trecho)]
                k_geom_local = matriz_rigidez_geometrica(N_comp, H_PAVIMENTO)
                K_geometrica[np.ix_(dofs, dofs)] += k_geom_local
        
        # A rigidez total do sistema cai proporcionalmente ao peso (K_E - K_G)
        K_total = K_elastica - K_geometrica
        K_ll = K_total[np.ix_(livres, livres)]
        
        try:
            condicao_matriz = float(np.linalg.cond(K_ll))
            desloc_livres_novos = np.linalg.solve(K_ll, F_l)
        except np.linalg.LinAlgError as exc:
            raise np.linalg.LinAlgError("A matriz de rigidez tornou-se singular (Colapso por Instabilidade).") from exc

        # Verificação de convergência
        erro_max = np.max(np.abs(desloc_livres_novos - desloc[livres]))
        desloc[livres] = desloc_livres_novos
        
        if erro_max < tolerancia:
            break
    else:
        # Se o loop terminar sem 'break', a estrutura não estabilizou
        raise ValueError(f"O cálculo P-Delta não convergiu após {max_iter} iterações. Estrutura altamente esbelta.")

    # 3. Pós-processamento dos esforços finais de 2ª ordem
    reacoes_globais = (K_elastica - K_geometrica) @ desloc - F
    elementos = {}
    x_diagrama = np.linspace(0.0, H_PAVIMENTO, 31)
    
    for coluna in range(N_COLUNAS):
        for trecho in range(n_pav):
            dofs = dofs_elementos[(coluna, trecho)]
            d_local = desloc[dofs]
            f_eq = cargas_equiv_elementos[(coluna, trecho)]
            N_comp = forcas_normais_estaticas[(coluna, trecho)]
            
            # Os esforços finais utilizam K_E e K_G locais combinadas
            k_geom_local = matriz_rigidez_geometrica(N_comp, H_PAVIMENTO)
            q_extremidade = (k_elastica_local - k_geom_local) @ d_local - f_eq
            q_h = peso_pilar_d * theta_imperfeicao_rad * sinal

            M_x = q_extremidade[1] - q_extremidade[0] * x_diagrama - 0.5 * q_h * x_diagrama**2

            elementos[(coluna, trecho)] = {
                "dofs": dofs,
                "deslocamentos_locais": d_local,
                "forcas_extremidade": q_extremidade,
                "x": x_diagrama.copy(),
                "M": M_x,
                "q_horizontal_imperfeicao": q_h,
            }

    desloc_niveis = np.array([desloc[nivel] for nivel in range(n_pav + 1)], dtype=float)
    rotacoes_base = np.array([desloc[idx_rotacao(c, 0, n_pav)] for c in range(N_COLUNAS)], dtype=float)

    return {
        "K": K_elastica,  # Retorna a elástica para referência, mas os esforços já são de 2ª ordem
        "F": F,
        "deslocamentos": desloc,
        "desloc_niveis": desloc_niveis,
        "reacoes_globais": reacoes_globais,
        "elementos": elementos,
        "condicao_matriz": condicao_matriz,
        "rotacoes_base": rotacoes_base,
        "vento_d": vento_d,
        "sinal": sinal,
    }

# =============================================================================
# 5. FORÇAS NORMAIS, DELTA M E GAMMA-Z
# =============================================================================

def calcular_normais_por_trecho(
    n_pav: int,
    reacoes_vigas: np.ndarray,
    peso_pilar_d: float,
) -> list[dict[str, float]]:
    """Força normal de compressão nos extremos de cada trecho de cada pilar."""
    resultados: list[dict[str, float]] = []
    for coluna in range(N_COLUNAS):
        for trecho in range(n_pav):
            nivel_superior = trecho + 1
            cargas_vigas_acima = float(np.sum(reacoes_vigas[nivel_superior:, coluna]))
            peso_trechos_acima = peso_pilar_d * H_PAVIMENTO * (n_pav - nivel_superior)
            N_topo = cargas_vigas_acima + peso_trechos_acima
            N_base = N_topo + peso_pilar_d * H_PAVIMENTO
            resultados.append(
                {
                    "Pilar": coluna + 1,
                    "Trecho": f"{trecho + 1}-{trecho + 2}",
                    "N_base_kN": N_base,
                    "N_topo_kN": N_topo,
                }
            )
    return resultados


def integrar_deslocamento_elemento(d_local: np.ndarray, comprimento: float) -> float:
    """Integra v(x) do elemento via quadratura de Gauss de 4 pontos."""
    pontos, pesos = np.polynomial.legendre.leggauss(4)
    integral = 0.0
    L = comprimento
    for xi, wi in zip(pontos, pesos):
        r = (xi + 1.0) / 2.0
        N1 = 1.0 - 3.0 * r**2 + 2.0 * r**3
        N2 = L * (r - 2.0 * r**2 + r**3)
        N3 = 3.0 * r**2 - 2.0 * r**3
        N4 = L * (-r**2 + r**3)
        v = N1 * d_local[0] + N2 * d_local[1] + N3 * d_local[2] + N4 * d_local[3]
        integral += wi * v * (L / 2.0)
    return float(integral)


def calcular_gamma_z(
    n_pav: int,
    resultado: dict[str, Any],
    reacoes_vigas: np.ndarray,
    peso_pilar_d: float,
    theta_imperfeicao_rad: float,
    vento_caracteristico: np.ndarray,
    fator_vento: float,
) -> dict[str, Any]:
    """Calcula M1, DeltaM e gamma_z; não força razão quando M1 = 0."""
    sinal = int(resultado["sinal"])
    cotas = H_PAVIMENTO * np.arange(1, n_pav + 1, dtype=float)

    # Momento global de primeira ordem das ações horizontais de cálculo.
    M1_vento = float(np.sum(resultado["vento_d"] * cotas))
    M1_imperfeicao_nodal = 0.0
    for nivel in range(1, n_pav + 1):
        p_nivel = float(np.sum(reacoes_vigas[nivel, :]))
        H_imp = sinal * theta_imperfeicao_rad * p_nivel
        M1_imperfeicao_nodal += H_imp * nivel * H_PAVIMENTO

    q_h_imp = sinal * theta_imperfeicao_rad * peso_pilar_d
    M1_imperfeicao_distribuida = 0.0
    for trecho in range(n_pav):
        z_centro = (trecho + 0.5) * H_PAVIMENTO
        M1_imperfeicao_distribuida += (
            N_COLUNAS * q_h_imp * H_PAVIMENTO * z_centro
        )

    M1_assinado = M1_vento + M1_imperfeicao_nodal + M1_imperfeicao_distribuida

    # DeltaM = soma P_i*delta_i nas reações de vigas + integral q_pilar*v(z) dz.
    desloc_niveis = resultado["desloc_niveis"]
    delta_nodal = 0.0
    for nivel in range(1, n_pav + 1):
        delta_nodal += float(np.sum(reacoes_vigas[nivel, :])) * desloc_niveis[nivel]

    delta_distribuido = 0.0
    for coluna in range(N_COLUNAS):
        for trecho in range(n_pav):
            elem = resultado["elementos"][(coluna, trecho)]
            integral_v = integrar_deslocamento_elemento(
                elem["deslocamentos_locais"], H_PAVIMENTO
            )
            delta_distribuido += peso_pilar_d * integral_v

    delta_assinado = delta_nodal + delta_distribuido
    if abs(M1_assinado) <= 1e-10:
        return {
            "M1_kNm": 0.0,
            "DeltaM_kNm": float("nan"),
            "razao": float("nan"),
            "gamma_z": None,
            "status": "Não aplicável: momento global de 1ª ordem nulo.",
            "delta_nodal": delta_nodal,
            "delta_distribuido": delta_distribuido,
        }

    # Projeta DeltaM no sentido do tombamento de primeira ordem.
    delta_destabilizante = float(np.sign(M1_assinado) * delta_assinado)
    M1_abs = abs(M1_assinado)
    razao = delta_destabilizante / M1_abs
    denominador = 1.0 - razao

    if denominador <= 0.0:
        gamma = None
        status = "Não finito: DeltaM/M1 >= 1; modelo indica perda de estabilidade." 
    elif razao < 0.0:
        gamma = 1.0 / denominador
        status = "Verificar sinal: DeltaM resultou estabilizante nesta combinação."
    else:
        gamma = 1.0 / denominador
        status = "Calculado pela expressão do enunciado; verificar domínio de aplicação."

    return {
        "M1_kNm": M1_abs,
        "DeltaM_kNm": delta_destabilizante,
        "razao": razao,
        "gamma_z": gamma,
        "status": status,
        "delta_nodal": delta_nodal,
        "delta_distribuido": delta_distribuido,
    }


# =============================================================================
# 6. HOGESLAG — PARÂMETROS EQUIVALENTES EXPLÍCITOS
# =============================================================================

# =============================================================================
# 6. HOGESLAG — AUTOMAÇÃO VIA QUOCIENTE DE RAYLEIGH
# =============================================================================

def calcular_hogeslag(
    F_vd_pavimentos: np.ndarray,
    n_pav: int,
    fundacao: str,
    fator_rigidez_hogeslag: float,
    
) -> dict[str, Any]:
    """
    Determina o multiplicador beta aplicando o Quociente de Rayleigh.
    Substitui a necessidade de arbitrar comprimentos de flambagem (l_e) e alturas (H_eq).
    """
    ndof = numero_graus_liberdade(n_pav)
    K_hog = np.zeros((ndof, ndof), dtype=float)
    EI_hog = EC * I_PILAR * fator_rigidez_hogeslag
    k_local = matriz_rigidez_flexao(EI_hog, H_PAVIMENTO)
    
    # Montagem da matriz de rigidez específica para o Hogeslag
    for coluna in range(N_COLUNAS):
        for trecho in range(n_pav):
            dofs = indices_globais(coluna, trecho, n_pav)
            K_hog[np.ix_(dofs, dofs)] += k_local
            
    if fundacao == "F2":
        for coluna in range(N_COLUNAS):
            dof_theta = idx_rotacao(coluna, 0, n_pav)
            K_hog[dof_theta, dof_theta] += K_ROTACIONAL_PILAR
            
    F_ficticio = np.zeros(ndof, dtype=float)
    fixos = [0] 
    if fundacao == "F1":
        fixos.extend(idx_rotacao(c, 0, n_pav) for c in range(N_COLUNAS))
    livres = np.array([i for i in range(ndof) if i not in set(fixos)], dtype=int)
    
    # Aplicação das forças horizontais fictícias iguais às cargas verticais de cálculo
    for nivel in range(1, n_pav + 1):
        F_ficticio[nivel] = F_vd_pavimentos[nivel - 1]
        
    K_ll = K_hog[np.ix_(livres, livres)]
    F_l = F_ficticio[livres]
    
    try:
        desloc_livres = np.linalg.solve(K_ll, F_l)
    except np.linalg.LinAlgError:
        return {"status": "Matriz singular (Instabilidade).", "beta": None, "gamma_hogeslag": None}
        
    desloc = np.zeros(ndof, dtype=float)
    desloc[livres] = desloc_livres
    desloc_translacao = np.array([desloc[nivel] for nivel in range(1, n_pav + 1)], dtype=float)
    
    # Quociente de Rayleigh para extração direta de beta
    numerador = float(np.sum(F_vd_pavimentos * desloc_translacao))
    denominador = float(np.sum(F_vd_pavimentos * desloc_translacao**2))
    
    beta = numerador / denominador if denominador > 0 else float("inf")
    carga_vertical_total = float(np.sum(F_vd_pavimentos))
    F_ref_kN = beta * carga_vertical_total
    
    if beta > 1.0:
        gamma_h = beta / (beta - 1.0)
        status = "Cálculo exato via Quociente de Rayleigh (dispensa l_e e H_eq)."
    else:
        gamma_h = None
        status = "beta <= 1: instabilidade indicada (Carga crítica superada)."

    return {
        "status": status,
        "F_ref_kN": F_ref_kN,
        "beta": beta,
        "gamma_hogeslag": gamma_h,
        "EI_eq_kNm2": fator_rigidez_hogeslag * EC * (N_COLUNAS * I_PILAR)
    }

# =============================================================================
# 7. CENÁRIOS E ENVOLTÓRIAS
# =============================================================================

def rotulo_sobrecarga(mascara: np.ndarray) -> str:
    niveis = []
    for i in range(mascara.shape[0]):
        vaos = [str(j + 1) for j in range(N_VAOS) if mascara[i, j]]
        if vaos:
            niveis.append(f"N{i + 1}:V{','.join(vaos)}")
    return "; ".join(niveis) if niveis else "Sem Q"


def executar_caso(
    n_pav: int,
    fundacao: str,
    combinacao: str,
    tipo_vento: str,
    mascara_q: np.ndarray,
    sinal: int,
    theta_imperfeicao_rad: float,
    fator_rigidez_primeira_ordem: float,
    fator_rigidez_hogeslag: float,
) -> dict[str, Any]:
    fatores = COMBINACOES[combinacao]
    reacoes, momentos, peso_pilar_d, carga_vertical_total = calcular_acoes_verticais(
        n_pav, mascara_q, combinacao
    )
    vento_k = distribuicao_vento(n_pav, tipo_vento)

    resultado = montar_e_resolver_modelo(
        n_pav=n_pav,
        fundacao=fundacao,
        reacoes_vigas=reacoes,
        momentos_exc=momentos,
        peso_pilar_d=peso_pilar_d,
        vento_caracteristico=vento_k,
        fator_vento=fatores["gamma_w"],
        sinal=sinal,
        theta_imperfeicao_rad=theta_imperfeicao_rad,
        fator_rigidez_primeira_ordem=fator_rigidez_primeira_ordem,
    )

    gamma = calcular_gamma_z(
        n_pav=n_pav,
        resultado=resultado,
        reacoes_vigas=reacoes,
        peso_pilar_d=peso_pilar_d,
        theta_imperfeicao_rad=theta_imperfeicao_rad,
        vento_caracteristico=vento_k,
        fator_vento=fatores["gamma_w"],
    )

    # Consolidação do vetor de forças verticais por pavimento para o Hogeslag
    F_vd_pav = np.zeros(n_pav, dtype=float)
    for nivel in range(1, n_pav + 1):
        # Soma as reações das vigas naquele nível e o peso próprio correspondente ao trecho dos 4 pilares
        F_vd_pav[nivel - 1] = float(np.sum(reacoes[nivel, :])) + (N_COLUNAS * peso_pilar_d * H_PAVIMENTO)

    hogeslag = calcular_hogeslag(
        F_vd_pavimentos=F_vd_pav,
        n_pav=n_pav,
        fundacao=fundacao,
        fator_rigidez_hogeslag=fator_rigidez_hogeslag,
    )

    normais = calcular_normais_por_trecho(n_pav, reacoes, peso_pilar_d)

    return {
        "estrutura_n_pav": n_pav,
        "fundacao": fundacao,
        "combinacao": combinacao,
        "vento": tipo_vento,
        "mascara_q": mascara_q.copy(),
        "rotulo_q": rotulo_sobrecarga(mascara_q),
        "sinal": sinal,
        "reacoes_vigas": reacoes,
        "momentos_exc": momentos,
        "peso_pilar_d": peso_pilar_d,
        "carga_vertical_total": carga_vertical_total,
        "resultado": resultado,
        "gamma": gamma,
        "hogeslag": hogeslag,
        "normais": normais,
    }


def gerar_mascaras_q(n_pav: int) -> list[np.ndarray]:
    """Todas as combinações binárias de carga Q nos 3 vãos de cada pavimento."""
    n_variaveis = n_pav * N_VAOS
    mascaras = []
    for valores in product([False, True], repeat=n_variaveis):
        mascaras.append(np.array(valores, dtype=bool).reshape(n_pav, N_VAOS))
    return mascaras


def registro_resumo(caso: dict[str, Any]) -> dict[str, Any]:
    gamma = caso["gamma"]
    hog = caso["hogeslag"]
    return {
        "Combinação": caso["combinacao"],
        "Vento": caso["vento"],
        "Sentido": "+" if caso["sinal"] > 0 else "−",
        "Sobrecarga": caso["rotulo_q"],
        "ΣFvd (kN)": caso["carga_vertical_total"],
        "M1 (kN.m)": gamma["M1_kNm"],
        "ΔM (kN.m)": gamma["DeltaM_kNm"],
        "ΔM/M1": gamma["razao"],
        "γz": gamma["gamma_z"],
        "β Hogeslag": hog.get("beta"),
        "γ Hogeslag": hog.get("gamma_hogeslag"),
        "Status γz": gamma["status"],
        "Status Hogeslag": hog["status"],
    }

# =============================================================================
# 8. INTERFACE STREAMLIT
# =============================================================================

st.set_page_config(page_title="Estabilidade — Hogeslag e γz", layout="wide")
st.title("Análise de estabilidade global — estruturas pré-moldadas")
st.caption(
    "Modelo de pilares contínuos com ligações viga-pilar articuladas, "
    "compatibilidade horizontal por nível e fundações F1/F2."
)

st.sidebar.header("1. Configuração estrutural")
estrutura = st.sidebar.selectbox(
    "Estrutura", ["E1 — 1 pavimento", "E2 — 2 pavimentos", "E3 — 3 pavimentos"]
)
n_pav = int(estrutura[1])
fundacao_label = st.sidebar.radio("Condição de fundação", ["F1 — Engaste", "F2 — Mola rotacional"])
fundacao = "F1" if fundacao_label.startswith("F1") else "F2"

st.sidebar.header("2. Ações e combinações")
combinacao_selecionada = st.sidebar.selectbox(
    "Combinação ELU", ["C1", "C2", "C3", "C4", "Todas"], index=1,
    format_func=lambda c: c if c == "Todas" else f"{c} — {COMBINACOES[c]['descricao']}"
)
tipo_vento = st.sidebar.radio("Distribuição do vento", ["V1", "V2"], horizontal=True)

st.sidebar.header("3. Imperfeição e rigidez")
theta_imp_mrad = st.sidebar.number_input(
    "Desaprumo global θ (mrad)", min_value=0.0, max_value=20.0, value=0.0, step=0.1,
    help="Informe o valor definido pelo critério normativo adotado. O programa não determina automaticamente θ."
)
theta_imp_rad = theta_imp_mrad / 1000.0
fator_rigidez_1ordem = st.sidebar.number_input(
    "Fator de rigidez EI para deslocamentos (1ª ordem)",
    min_value=0.01, max_value=1.0, value=1.0, step=0.05,
    help="1,0 usa a rigidez bruta Ec.I. Reduções devem ser justificadas pelo modelo/referência adotados."
)

st.sidebar.header("4. Parâmetros equivalentes de Hogeslag")
fator_rigidez_hog = st.sidebar.number_input(
    "Fator aplicado a (EI)eq de Hogeslag", min_value=0.01, max_value=1.0,
    value=1.0, step=0.05,
    help="Não se presume EI/3. Informe uma redução apenas quando ela estiver justificada."
)
# Os inputs manuais de le_hogeslag e h_eq_hogeslag foram eliminados graças ao Quociente de Rayleigh

st.markdown("### Distribuição da sobrecarga Qk")
st.write(
    "Marque independentemente cada vão (V1, V2 e V3) em cada pavimento. "
    "A reação e o momento excêntrico são calculados separadamente para cada apoio."
)
modo_q = st.radio(
    "Modo de carregamento variável",
    ["Definir manualmente", "Varredura de todas as distribuições Q"],
    horizontal=True,
)

mascara_manual = np.zeros((n_pav, N_VAOS), dtype=bool)
if modo_q == "Definir manualmente":
    for nivel in range(n_pav):
        with st.expander(f"Pavimento {nivel + 1} — cota {(nivel + 1) * H_PAVIMENTO:.0f} m", expanded=True):
            cols = st.columns(3)
            for vao in range(N_VAOS):
                mascara_manual[nivel, vao] = cols[vao].checkbox(
                    f"Sobrecarga no vão {vao + 1}",
                    value=True,
                    key=f"q_{n_pav}_{nivel}_{vao}",
                )
else:
    st.info(
        f"Serão avaliadas {2 ** (3 * n_pav)} distribuições de Q "
        f"(incluindo ausência de sobrecarga), com os sentidos horizontais aplicáveis."
    )

st.markdown("---")
executar = st.button("Executar análise", type="primary", use_container_width=True)

if executar:
    mascaras = [mascara_manual] if modo_q == "Definir manualmente" else gerar_mascaras_q(n_pav)
    combinacoes = list(COMBINACOES.keys()) if combinacao_selecionada == "Todas" else [combinacao_selecionada]

    casos: list[dict[str, Any]] = []
    erros: list[str] = []

    barra = st.progress(0.0, text="Montando cenários...")
    total = len(mascaras) * sum(2 if COMBINACOES[c]["gamma_w"] > 0.0 or theta_imp_rad > 0.0 else 1 for c in combinacoes)
    feito = 0

    for mascara in mascaras:
        for combinacao in combinacoes:
            fatores = COMBINACOES[combinacao]
            sinais = [1, -1] if (fatores["gamma_w"] > 0.0 or theta_imp_rad > 0.0) else [1]
            for sinal in sinais:
                feito += 1
                try:
                    caso = executar_caso(
                        n_pav=n_pav,
                        fundacao=fundacao,
                        combinacao=combinacao,
                        tipo_vento=tipo_vento,
                        mascara_q=mascara,
                        sinal=sinal,
                        theta_imperfeicao_rad=theta_imp_rad,
                        fator_rigidez_primeira_ordem=fator_rigidez_1ordem,
                        fator_rigidez_hogeslag=fator_rigidez_hog,
                    )
                    casos.append(caso)
                except (np.linalg.LinAlgError, ValueError, FloatingPointError) as exc:
                    erros.append(f"{combinacao} / {rotulo_sobrecarga(mascara)} / sinal {sinal}: {exc}")
                barra.progress(min(feito / max(total, 1), 1.0), text=f"Analisando cenário {feito}/{total}")

    barra.empty()

    if not casos:
        st.error("Nenhum cenário pôde ser resolvido.")
        for erro in erros[:20]:
            st.write(f"- {erro}")
        st.stop()

    df = pd.DataFrame([registro_resumo(caso) for caso in casos])

    # Escolhe o caso governante para exibição detalhada: maior razão ΔM/M1 válida.
    df_validos = df[np.isfinite(pd.to_numeric(df["ΔM/M1"], errors="coerce"))]
    if not df_validos.empty:
        indice_critico = pd.to_numeric(df_validos["ΔM/M1"]).idxmax()
        caso_critico = casos[int(indice_critico)]
    else:
        caso_critico = casos[0]

    st.success(f"Análise concluída: {len(casos)} casos resolvidos.")
    if erros:
        st.warning(f"{len(erros)} cenários não puderam ser resolvidos. Primeiro(s) aviso(s):")
        for erro in erros[:10]:
            st.write(f"- {erro}")

    st.header("1. Resumo dos cenários")
    st.dataframe(df, use_container_width=True, hide_index=True)
    csv = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "Baixar resumo completo em CSV", data=csv,
        file_name=f"resumo_estabilidade_E{n_pav}_{fundacao}.csv", mime="text/csv"
    )

    st.header("2. Caso governante para detalhamento")
    st.caption(
        f"Combinação {caso_critico['combinacao']} | vento {caso_critico['vento']} | "
        f"sentido {'+' if caso_critico['sinal'] > 0 else '−'} | {caso_critico['rotulo_q']}"
    )

    gamma = caso_critico["gamma"]
    hog = caso_critico["hogeslag"]
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("M1,tot,d", f"{gamma['M1_kNm']:.2f} kN.m" if np.isfinite(gamma["M1_kNm"]) else "—")
    r2.metric("ΔMtot,d", f"{gamma['DeltaM_kNm']:.2f} kN.m" if np.isfinite(gamma["DeltaM_kNm"]) else "—")
    r3.metric("γz", f"{gamma['gamma_z']:.4f}" if gamma["gamma_z"] is not None else "Não definido")
    r4.metric("ΣFvd", f"{caso_critico['carga_vertical_total']:.2f} kN")
    st.info(gamma["status"])

    with st.expander("Detalhamento de Hogeslag (Quociente de Rayleigh)", expanded=True):
        st.write(hog["status"])
        if hog.get("beta") is not None:
            h1, h2 = st.columns(2)
            h1.metric("Fref (Carga Crítica de Rayleigh)", f"{hog['F_ref_kN']:.2f} kN")
            h2.metric("β (Fator de Segurança)", f"{hog['beta']:.4f}")
            st.write(f"Rigidez equivalente utilizada: **{hog['EI_eq_kNm2']:.2f} kN.m²**.")
            st.write(
                f"Multiplicador de Hogeslag: **{hog['gamma_hogeslag']:.4f}**"
                if hog["gamma_hogeslag"] is not None else "Multiplicador de Hogeslag não finito."
            )

    st.header("3. Deslocamentos por nível (Amplificados pela P-Δ)")
    desloc = caso_critico["resultado"]["desloc_niveis"]
    df_desloc = pd.DataFrame(
        {
            "Nível": list(range(n_pav + 1)),
            "Cota (m)": [i * H_PAVIMENTO for i in range(n_pav + 1)],
            "Deslocamento (mm)": desloc * 1000.0,
        }
    )
    st.dataframe(df_desloc, use_container_width=True, hide_index=True)
    st.line_chart(df_desloc.set_index("Cota (m)")["Deslocamento (mm)"])

    st.header("4. Reações verticais, momentos excêntricos e forças normais")
    linhas_reacoes = []
    for nivel in range(1, n_pav + 1):
        for coluna in range(N_COLUNAS):
            linhas_reacoes.append(
                {
                    "Nível": nivel,
                    "Cota (m)": nivel * H_PAVIMENTO,
                    "Pilar": coluna + 1,
                    "Reação vertical das vigas (kN)": caso_critico["reacoes_vigas"][nivel, coluna],
                    "Momento excêntrico (kN.m)": caso_critico["momentos_exc"][nivel, coluna],
                }
            )
    st.dataframe(pd.DataFrame(linhas_reacoes), use_container_width=True, hide_index=True)
    st.subheader("Força normal nos trechos dos pilares")
    st.dataframe(pd.DataFrame(caso_critico["normais"]), use_container_width=True, hide_index=True)

    st.header("5. Diagramas de momento fletor dos pilares")
    elem_res = caso_critico["resultado"]["elementos"]
    for coluna in range(N_COLUNAS):
        with st.expander(f"Pilar {coluna + 1}", expanded=(coluna == 0)):
            dados_grafico = {}
            for trecho in range(n_pav):
                item = elem_res[(coluna, trecho)]
                dados_grafico[f"Trecho {trecho + 1}-{trecho + 2}"] = item["M"]
                st.write(
                    f"Trecho {trecho + 1}-{trecho + 2}: "
                    f"Mbase = {item['M'][0]:.2f} kN.m; "
                    f"Mtopo = {item['M'][-1]:.2f} kN.m"
                )
            df_m = pd.DataFrame(dados_grafico, index=elem_res[(coluna, 0)]["x"])
            df_m.index.name = "Coordenada local z (m)"
            st.line_chart(df_m)

    st.header("6. Envoltórias considerando os cenários analisados")
    st.caption(
        "As envoltórias abaixo abrangem os sentidos horizontais e as distribuições de Q "
        "que foram efetivamente incluídos nesta execução. Para a envoltória completa de Q, "
        "utilize a opção 'Varredura de todas as distribuições Q'."
    )
    linhas_env = []
    for coluna in range(N_COLUNAS):
        for trecho in range(n_pav):
            chave = (coluna, trecho)
            pilhas = np.vstack([c["resultado"]["elementos"][chave]["M"] for c in casos])
            x = casos[0]["resultado"]["elementos"][chave]["x"]
            m_min = np.min(pilhas, axis=0)
            m_max = np.max(pilhas, axis=0)
            for xi, mn, mx in zip(x, m_min, m_max):
                linhas_env.append(
                    {
                        "Pilar": coluna + 1,
                        "Trecho": f"{trecho + 1}-{trecho + 2}",
                        "z local (m)": xi,
                        "Mmin (kN.m)": mn,
                        "Mmax (kN.m)": mx,
                    }
                )
            with st.expander(f"Envoltória — Pilar {coluna + 1}, trecho {trecho + 1}-{trecho + 2}"):
                df_env = pd.DataFrame(
                    {
                        "Mmin": m_min,
                        "Mmax": m_max,
                    }, index=x
                )
                df_env.index.name = "z local (m)"
                st.line_chart(df_env)

    st.download_button(
        "Baixar pontos das envoltórias em CSV",
        data=pd.DataFrame(linhas_env).to_csv(index=False).encode("utf-8-sig"),
        file_name=f"envoltorias_momentos_E{n_pav}_{fundacao}.csv",
        mime="text/csv",
    )

    st.header("7. Reações de apoio e verificações numéricas")
    res_global = caso_critico["resultado"]["reacoes_globais"]
    reacao_horizontal_base = float(res_global[0])
    st.write(f"Reação horizontal total na base: **{reacao_horizontal_base:.2f} kN**.")

    linhas_base = []
    for coluna in range(N_COLUNAS):
        q_base = caso_critico["resultado"]["elementos"][(coluna, 0)]["forcas_extremidade"]
        linha = {
            "Pilar": coluna + 1,
            "Cortante na base (kN)": q_base[0],
            "Momento de extremidade na base (kN.m)": q_base[1],
            "Força normal na base (kN)": next(
                item["N_base_kN"] for item in caso_critico["normais"]
                if item["Pilar"] == coluna + 1 and item["Trecho"] == "1-2"
            ),
        }
        if fundacao == "F2":
            theta_base = caso_critico["resultado"]["rotacoes_base"][coluna]
            linha["Momento da mola (kN.m)"] = K_ROTACIONAL_PILAR * theta_base
        linhas_base.append(linha)
    st.dataframe(pd.DataFrame(linhas_base), use_container_width=True, hide_index=True)

    if fundacao == "F2":
        st.caption(
            "Em F2, a rotação da base é livre e o momento de reação da fundação é "
            "fornecido pela mola rotacional de cada pilar."
        )
    st.write(
        f"Número de condição da matriz reduzida: "
        f"**{caso_critico['resultado']['condicao_matriz']:.3e}**. "
        "Valores muito elevados podem indicar mal condicionamento numérico."
    )

    st.warning(
        "O cálculo iterativo P-Δ de segunda ordem geométrica foi implementado para a determinação dos "
        "deslocamentos e diagramas de momento definitivos. O fator γz exposto é uma métrica normativa comparativa."
    )

    with st.expander("Hipóteses e limitações que devem constar no relatório"):
        st.markdown(
            """
            1. As vigas são consideradas axialmente indeformáveis para impor uma translação horizontal comum aos quatro pilares; suas rotações não são acopladas aos pilares.
            2. Cada pilar possui rotação nodal própria. Os quatro pilares não são substituídos por uma única barra com `4EI` na análise matricial.
            3. As reações de cada vão são calculadas individualmente. O momento excêntrico no pilar interno é a soma algébrica das contribuições dos vãos adjacentes.
            4. O peso próprio dos pilares é aplicado como carga vertical distribuída. Sua contribuição em ΔM é integrada ao longo de cada trecho (quadratura de Gauss).
            5. A matriz de rigidez geométrica iterativa garante a convergência P-Δ para efeitos globais de 2ª ordem reais, independentes dos multiplicadores.
            6. O método de Hogeslag encontra a carga crítica via Quociente de Rayleigh através de forças horizontais fictícias baseadas nos pesos dos pavimentos, prescindindo de estipular um `l_e` equivalente global.
            7. O desaprumo é convertido em ações horizontais equivalentes `P.θ`.
            8. Os diagramas de momento refletem o equilíbrio da configuração deformada e da carga horizontal distribuída da imperfeição.
            """
        )
else:
    st.info("Configure os parâmetros e clique em **Executar análise** para gerar os resultados.")

st.markdown("---")
st.caption(
    "Ferramenta académica em desenvolvimento. Conferir as hipóteses, unidades, sinais e "
    "critérios normativos antes de utilizar os resultados em qualquer conclusão de projeto."
)