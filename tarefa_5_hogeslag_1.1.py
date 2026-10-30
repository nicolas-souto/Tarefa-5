# -*- coding: utf-8 -*-
"""
Análise de estabilidade global de estruturas pré-moldadas — Hogeslag e gamma_z

Aplicativo acadêmico em Streamlit, desenvolvido a partir do enunciado fornecido.

MODELO E HIPÓTESES
------------------
- Pórtico plano com 4 pilares contínuos e 3 vãos de 18 m em cada nível.
- Pé-direito de 6 m; pilares com seção quadrada configurável.
- Ligações viga-pilar articuladas: as vigas não fornecem rigidez rotacional aos pilares.
- Vigas axialmente indeformáveis: uma translação horizontal comum por nível.
- Cada pilar tem rotações próprias, contínuas entre elementos adjacentes.
- Elementos de flexão de Euler-Bernoulli.
- F1: translação e rotação da base impedidas.
- F2: translação da base impedida e mola rotacional de 75 MN.m/rad por pilar.
- Reações de vigas e momentos excêntricos montados vão a vão, com sinais algébricos.
- Peso próprio dos pilares aplicado como carga vertical distribuída. Seu valor é recalculado
  pela seção e pelo peso específico do concreto (25 kN/m³); para 40 x 40 cm resulta em 4 kN/m.
- Imperfeição global representada por ações horizontais equivalentes P*theta, incluindo
  as cargas concentradas nos níveis e o peso próprio distribuído dos pilares.

ANÁLISES
--------
1) Primeira ordem: K_E d = F. É esta solução que alimenta o cálculo didático de gamma_z.
2) Segunda ordem geométrica linearizada: (K_E - K_G) d = F, com esforços normais de
   primeira ordem mantidos constantes. Não é uma análise geometricamente não linear iterativa.
3) Flambagem elástica por autovalores: referência complementar independente, não chamada
   de método de Hogeslag.
4) Hogeslag: expressões do enunciado (F_e, F_f, F_ref, beta e gamma). Como o enunciado
   alerta que a equivalência de carga no topo não se aplica automaticamente a vários níveis,
   le/H_total e H_f são parâmetros explícitos/hipóteses declaradas pelo usuário.

LIMITAÇÕES IMPORTANTES
----------------------
- gamma_z é apresentado como comparação didática: E1, E2 e E3 têm menos de quatro pavimentos.
- A análise de segunda ordem é uma formulação elástica linearizada com K_G montada com N de
  primeira ordem; não inclui não linearidade material, grandes rotações, efeitos locais de
  segunda ordem de seção ou redistribuição não linear das forças normais.
- Os parâmetros equivalentes de Hogeslag devem ser justificados com a bibliografia da disciplina.
- A recuperação de momentos é feita pelo equilíbrio do elemento, usando forças de extremidade
  e a contribuição geométrica N*v'. Deve ser validada com um caso de referência independente.
- Não substitui projeto, verificação normativa completa ou análise profissional.

EXECUÇÃO
--------
py -3.13 -m pip install streamlit numpy pandas
py -3.13 -m streamlit run tarefa_5_hogeslag_revisado.py
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import pi
from typing import Any, Iterable

import numpy as np
import pandas as pd

try:
    import streamlit as st
except ModuleNotFoundError:  # Permite testar o núcleo numérico sem instalar Streamlit.
    st = None


# =============================================================================
# 1. DADOS DO ENUNCIADO E PARÂMETROS
# =============================================================================

N_PILARES = 4
N_VAOS = 3
H_PAVIMENTO_PADRAO = 6.0
L_VAO_PADRAO = 18.0
EC_PADRAO = 30.0e6                 # kN/m² = 30 GPa
GAMMA_CONCRETO_PADRAO = 25.0      # kN/m³
G_K_VIGA_PADRAO = 13.0            # kN/m
Q_K_VIGA_PADRAO = 7.0             # kN/m
EXCENTRICIDADE_PADRAO = 0.40      # m
K_ROTACIONAL_PILAR = 75_000.0     # kN.m/rad por pilar

COMBINACOES: dict[str, dict[str, Any]] = {
    "C1": {"gamma_g": 1.40, "gamma_q": 1.40, "gamma_w": 0.00,
           "descricao": "Q principal, sem vento"},
    "C2": {"gamma_g": 1.40, "gamma_q": 1.40, "gamma_w": 0.84,
           "descricao": "Q principal, vento acompanhante (psi0,W = 0,6)"},
    "C3": {"gamma_g": 1.40, "gamma_q": 1.12, "gamma_w": 1.40,
           "descricao": "Vento principal, Q acompanhante (psi0,Q = 0,8)"},
    "C4": {"gamma_g": 1.40, "gamma_q": 0.00, "gamma_w": 1.40,
           "descricao": "Vento principal, sem Q"},
}


@dataclass(frozen=True)
class ParametrosModelo:
    n_pav: int
    lado_pilar: float = 0.40
    ec: float = EC_PADRAO
    gamma_concreto: float = GAMMA_CONCRETO_PADRAO
    h_pavimento: float = H_PAVIMENTO_PADRAO
    l_vao: float = L_VAO_PADRAO
    gk_viga: float = G_K_VIGA_PADRAO
    qk_viga: float = Q_K_VIGA_PADRAO
    excentricidade: float = EXCENTRICIDADE_PADRAO
    fator_ei_1ordem: float = 1.0
    fator_ei_hogeslag: float = 1.0

    @property
    def area_pilar(self) -> float:
        return self.lado_pilar**2

    @property
    def inercia_pilar(self) -> float:
        return self.lado_pilar**4 / 12.0

    @property
    def peso_proprio_pilar(self) -> float:
        return self.gamma_concreto * self.area_pilar

    @property
    def altura_total(self) -> float:
        return self.n_pav * self.h_pavimento

    @property
    def ei_1ordem(self) -> float:
        return self.ec * self.inercia_pilar * self.fator_ei_1ordem

    @property
    def ei_hogeslag_total(self) -> float:
        # Quatro pilares em paralelo sob a hipótese de translação horizontal comum.
        return N_PILARES * self.ec * self.inercia_pilar * self.fator_ei_hogeslag


# =============================================================================
# 2. MATRIZES E FUNÇÕES DE ELEMENTO
# =============================================================================

def matriz_rigidez_flexao(ei: float, comprimento: float) -> np.ndarray:
    """Matriz local Euler-Bernoulli para [u_i, theta_i, u_j, theta_j]."""
    L = float(comprimento)
    return (ei / L**3) * np.array(
        [
            [12.0, 6.0 * L, -12.0, 6.0 * L],
            [6.0 * L, 4.0 * L**2, -6.0 * L, 2.0 * L**2],
            [-12.0, -6.0 * L, 12.0, -6.0 * L],
            [6.0 * L, 2.0 * L**2, -6.0 * L, 4.0 * L**2],
        ], dtype=float,
    )


def matriz_rigidez_geometrica(n_compressao: float, comprimento: float) -> np.ndarray:
    """Matriz geométrica consistente para compressão positiva."""
    L = float(comprimento)
    N = max(float(n_compressao), 0.0)
    return (N / (30.0 * L)) * np.array(
        [
            [36.0, 3.0 * L, -36.0, 3.0 * L],
            [3.0 * L, 4.0 * L**2, -3.0 * L, -L**2],
            [-36.0, -3.0 * L, 36.0, -3.0 * L],
            [3.0 * L, -L**2, -3.0 * L, 4.0 * L**2],
        ], dtype=float,
    )


def vetor_carga_distribuida(q: float, comprimento: float) -> np.ndarray:
    """Vetor nodal consistente para carga transversal uniforme positiva em +u."""
    L = float(comprimento)
    return float(q) * np.array([L / 2.0, L**2 / 12.0, L / 2.0, -L**2 / 12.0])


def matriz_rigidez_geometrica_variavel(
    n_base: float, n_topo: float, comprimento: float, n_gauss: int = 5
) -> np.ndarray:
    """Integra K_G = integral N(x) B(x)^T B(x) dx para N linear no trecho.

    N_base e N_topo são compressões positivas nos extremos. O peso próprio distribuído
    faz N variar linearmente ao longo do pilar; a quadratura de Gauss integra exatamente
    o produto polinomial resultante para a interpolação cúbica usada.
    """
    L = float(comprimento)
    pontos, pesos = np.polynomial.legendre.leggauss(n_gauss)
    x = 0.5 * L * (pontos + 1.0)
    w = 0.5 * L * pesos
    N_x = float(n_base) + (float(n_topo) - float(n_base)) * (x / L)
    _, dN = funcao_forma_hermite(x, L)
    return np.einsum("g,gi,gj->ij", w * N_x, dN, dN)


def numero_graus_liberdade(n_pav: int) -> int:
    n_niveis = n_pav + 1
    return (N_PILARES + 1) * n_niveis


def idx_u(nivel: int) -> int:
    """Índice da translação horizontal comum ao nível."""
    return int(nivel)


def idx_theta(coluna: int, nivel: int, n_pav: int) -> int:
    n_niveis = n_pav + 1
    return n_niveis + coluna * n_niveis + nivel


def dofs_elemento(coluna: int, trecho: int, n_pav: int) -> list[int]:
    return [
        idx_u(trecho),
        idx_theta(coluna, trecho, n_pav),
        idx_u(trecho + 1),
        idx_theta(coluna, trecho + 1, n_pav),
    ]


def funcao_forma_hermite(x: float | np.ndarray, L: float) -> tuple[np.ndarray, np.ndarray]:
    """Retorna N(x) e dN/dx para interpolação cúbica de deslocamentos."""
    r = np.asarray(x, dtype=float) / L
    n = np.stack(
        [
            1.0 - 3.0 * r**2 + 2.0 * r**3,
            L * (r - 2.0 * r**2 + r**3),
            3.0 * r**2 - 2.0 * r**3,
            L * (-r**2 + r**3),
        ], axis=-1,
    )
    dn_dr = np.stack(
        [
            -6.0 * r + 6.0 * r**2,
            L * (1.0 - 4.0 * r + 3.0 * r**2),
            6.0 * r - 6.0 * r**2,
            L * (-2.0 * r + 3.0 * r**2),
        ], axis=-1,
    )
    return n, dn_dr / L


def deslocamento_elemento(d_local: np.ndarray, x: float | np.ndarray, L: float) -> np.ndarray:
    N, _ = funcao_forma_hermite(x, L)
    return np.sum(N * d_local, axis=-1)


def inclinacao_elemento(d_local: np.ndarray, x: float | np.ndarray, L: float) -> np.ndarray:
    _, dN = funcao_forma_hermite(x, L)
    return np.sum(dN * d_local, axis=-1)


# =============================================================================
# 3. AÇÕES VERTICAIS, REAÇÕES E MOMENTOS EXCÊNTRICOS
# =============================================================================

def calcular_acoes_verticais(
    params: ParametrosModelo,
    mascara_q: np.ndarray,
    combinacao: str,
) -> dict[str, Any]:
    """Calcula reações de cada vão, momentos excêntricos e peso próprio dos pilares."""
    if mascara_q.shape != (params.n_pav, N_VAOS):
        raise ValueError("A máscara de Q deve ter dimensão (número de pavimentos, 3 vãos).")

    fatores = COMBINACOES[combinacao]
    g_d = fatores["gamma_g"] * params.gk_viga
    q_d = fatores["gamma_q"] * params.qk_viga
    reacoes = np.zeros((params.n_pav + 1, N_PILARES), dtype=float)
    momentos = np.zeros_like(reacoes)

    for nivel in range(1, params.n_pav + 1):
        for vao in range(N_VAOS):
            carga_linear = g_d + q_d * float(mascara_q[nivel - 1, vao])
            reacao = carga_linear * params.l_vao / 2.0
            p_esq, p_dir = vao, vao + 1
            reacoes[nivel, p_esq] += reacao
            reacoes[nivel, p_dir] += reacao
            # Convenção do enunciado/modelo: reação do vão à direita = -eR;
            # reação do vão à esquerda = +eR.
            momentos[nivel, p_esq] -= params.excentricidade * reacao
            momentos[nivel, p_dir] += params.excentricidade * reacao

    gamma_g = fatores["gamma_g"]
    peso_pilar_d = gamma_g * params.peso_proprio_pilar
    peso_total_pilares = N_PILARES * params.n_pav * params.h_pavimento * peso_pilar_d
    carga_vigas_total = float(np.sum(reacoes))
    carga_total = carga_vigas_total + peso_total_pilares

    # Cargas verticais por nível para o centroide de aplicação: vigas no nível;
    # peso dos pilares de cada trecho no meio do respectivo trecho.
    cargas_niveis = np.sum(reacoes[1:, :], axis=1)
    z_niveis = params.h_pavimento * np.arange(1, params.n_pav + 1, dtype=float)
    momento_estatico = float(np.sum(cargas_niveis * z_niveis))
    peso_por_trecho = N_PILARES * peso_pilar_d * params.h_pavimento
    z_centros = params.h_pavimento * (np.arange(params.n_pav, dtype=float) + 0.5)
    momento_estatico += float(np.sum(peso_por_trecho * z_centros))
    h_eq_centroide = momento_estatico / carga_total if carga_total > 0.0 else float("nan")

    return {
        "reacoes": reacoes,
        "momentos_exc": momentos,
        "peso_pilar_d_kN_m": peso_pilar_d,
        "carga_vigas_total_kN": carga_vigas_total,
        "peso_pilares_total_kN": peso_total_pilares,
        "carga_vertical_total_kN": carga_total,
        "cargas_vigas_nivel_kN": cargas_niveis,
        "h_eq_centroide_m": h_eq_centroide,
    }


def distribuicao_vento(params: ParametrosModelo, tipo: str) -> np.ndarray:
    cotas = params.h_pavimento * np.arange(1, params.n_pav + 1, dtype=float)
    if tipo == "V1":
        return np.full(params.n_pav, 80.0, dtype=float)
    if tipo == "V2":
        return 80.0 * cotas / float(np.sum(cotas))
    raise ValueError(f"Tipo de vento desconhecido: {tipo}")


def normais_elementos(
    params: ParametrosModelo,
    reacoes: np.ndarray,
    peso_pilar_d: float,
) -> dict[tuple[int, int], dict[str, float]]:
    """Força normal nos extremos; N varia linearmente pelo peso próprio do trecho."""
    saida: dict[tuple[int, int], dict[str, float]] = {}
    h = params.h_pavimento
    for coluna in range(N_PILARES):
        for trecho in range(params.n_pav):
            nivel_sup = trecho + 1
            cargas_acima = float(np.sum(reacoes[nivel_sup:, coluna]))
            peso_acima = peso_pilar_d * h * (params.n_pav - nivel_sup)
            n_topo = cargas_acima + peso_acima
            n_base = n_topo + peso_pilar_d * h
            saida[(coluna, trecho)] = {
                "N_base_kN": n_base,
                "N_topo_kN": n_topo,
                "N_medio_kN": 0.5 * (n_base + n_topo),
            }
    return saida


# =============================================================================
# 4. MONTAGEM DO MODELO E SOLUÇÃO DE 1ª / 2ª ORDEM
# =============================================================================

def montar_matriz_elastica(
    params: ParametrosModelo,
    fundacao: str,
) -> tuple[np.ndarray, np.ndarray, dict[tuple[int, int], list[int]]]:
    ndof = numero_graus_liberdade(params.n_pav)
    K = np.zeros((ndof, ndof), dtype=float)
    k_local = matriz_rigidez_flexao(params.ei_1ordem, params.h_pavimento)
    dofs_map: dict[tuple[int, int], list[int]] = {}
    for coluna in range(N_PILARES):
        for trecho in range(params.n_pav):
            dofs = dofs_elemento(coluna, trecho, params.n_pav)
            dofs_map[(coluna, trecho)] = dofs
            K[np.ix_(dofs, dofs)] += k_local

    if fundacao == "F2":
        for coluna in range(N_PILARES):
            K[idx_theta(coluna, 0, params.n_pav), idx_theta(coluna, 0, params.n_pav)] += K_ROTACIONAL_PILAR
    elif fundacao != "F1":
        raise ValueError("Fundação deve ser F1 ou F2.")
    return K, k_local, dofs_map


def graus_livres(params: ParametrosModelo, fundacao: str) -> tuple[np.ndarray, list[int]]:
    fixos = [idx_u(0)]
    if fundacao == "F1":
        fixos.extend(idx_theta(c, 0, params.n_pav) for c in range(N_PILARES))
    livres = np.array([i for i in range(numero_graus_liberdade(params.n_pav)) if i not in set(fixos)], dtype=int)
    return livres, fixos


def verificar_positividade(K_reduzida: np.ndarray, tolerancia_relativa: float = 1e-10) -> dict[str, Any]:
    """Diagnostica positividade da matriz simétrica reduzida."""
    Ksim = 0.5 * (K_reduzida + K_reduzida.T)
    autovalores = np.linalg.eigvalsh(Ksim)
    menor = float(np.min(autovalores))
    maior = float(np.max(autovalores))
    limite = tolerancia_relativa * max(abs(maior), 1.0)
    positiva = menor > limite
    condicao = float(maior / menor) if positiva else float("inf")
    return {
        "positiva_definida": positiva,
        "lambda_min": menor,
        "lambda_max": maior,
        "condicao": condicao,
        "autovalores": autovalores,
        "limite_tolerancia": limite,
    }


def resolver_sistema(
    K: np.ndarray,
    F: np.ndarray,
    params: ParametrosModelo,
    fundacao: str,
) -> dict[str, Any]:
    livres, fixos = graus_livres(params, fundacao)
    Kll = K[np.ix_(livres, livres)]
    Fl = F[livres]
    diagnostico = verificar_positividade(Kll)
    if not diagnostico["positiva_definida"]:
        raise np.linalg.LinAlgError(
            "Matriz reduzida sem positividade definida: "
            f"lambda_min={diagnostico['lambda_min']:.4e}, "
            f"lambda_max={diagnostico['lambda_max']:.4e}. "
            "A configuração não será tratada como equilíbrio estável."
        )
    dl = np.linalg.solve(Kll, Fl)
    d = np.zeros(K.shape[0], dtype=float)
    d[livres] = dl
    reacoes = K @ d - F
    diagnostico["graus_livres"] = livres
    diagnostico["graus_fixos"] = fixos
    return {"d": d, "reacoes": reacoes, "diagnostico": diagnostico}


def montar_vetor_forcas(
    params: ParametrosModelo,
    fundacao: str,
    acoes: dict[str, Any],
    vento_k: np.ndarray,
    fator_vento: float,
    sinal_vento: int,
    theta_imp_rad: float,
    sinal_imperfeicao: int,
) -> tuple[np.ndarray, dict[tuple[int, int], np.ndarray], dict[tuple[int, int], float]]:
    """Monta vento, momentos excêntricos e ações equivalentes de imperfeição."""
    F = np.zeros(numero_graus_liberdade(params.n_pav), dtype=float)
    cargas_eq: dict[tuple[int, int], np.ndarray] = {}
    q_imp_elementos: dict[tuple[int, int], float] = {}
    reacoes = acoes["reacoes"]
    peso_pilar_d = float(acoes["peso_pilar_d_kN_m"])
    h = params.h_pavimento

    # Imperfeição do peso próprio dos pilares: q_h = w_d * theta.
    q_h = peso_pilar_d * theta_imp_rad * sinal_imperfeicao
    for coluna in range(N_PILARES):
        for trecho in range(params.n_pav):
            dofs = dofs_elemento(coluna, trecho, params.n_pav)
            f_eq = vetor_carga_distribuida(q_h, h)
            F[dofs] += f_eq
            cargas_eq[(coluna, trecho)] = f_eq
            q_imp_elementos[(coluna, trecho)] = q_h

    # Vento e imperfeição P*theta das reações das vigas em cada pavimento.
    vento_d = fator_vento * vento_k * sinal_vento
    for nivel in range(1, params.n_pav + 1):
        P_vigas_nivel = float(np.sum(reacoes[nivel, :]))
        H_imp = theta_imp_rad * sinal_imperfeicao * P_vigas_nivel
        F[idx_u(nivel)] += float(vento_d[nivel - 1]) + H_imp
        for coluna in range(N_PILARES):
            F[idx_theta(coluna, nivel, params.n_pav)] += acoes["momentos_exc"][nivel, coluna]

    return F, cargas_eq, q_imp_elementos


def montar_matriz_geometrica(
    params: ParametrosModelo,
    dofs_map: dict[tuple[int, int], list[int]],
    normais: dict[tuple[int, int], dict[str, float]],
) -> np.ndarray:
    KG = np.zeros((numero_graus_liberdade(params.n_pav),) * 2, dtype=float)
    for chave, dofs in dofs_map.items():
        kg_local = matriz_rigidez_geometrica_variavel(
            normais[chave]["N_base_kN"],
            normais[chave]["N_topo_kN"],
            params.h_pavimento,
        )
        KG[np.ix_(dofs, dofs)] += kg_local
    return KG


def resolver_analises_estruturais(
    params: ParametrosModelo,
    fundacao: str,
    acoes: dict[str, Any],
    vento_k: np.ndarray,
    fator_vento: float,
    sinal_vento: int,
    theta_imp_rad: float,
    sinal_imperfeicao: int,
    calcular_diagramas: bool = False,
) -> dict[str, Any]:
    """Resolve 1ª ordem e, se estável, 2ª ordem geométrica linearizada."""
    K_E, k_local, dofs_map = montar_matriz_elastica(params, fundacao)
    F, cargas_eq, q_imp = montar_vetor_forcas(
        params, fundacao, acoes, vento_k, fator_vento, sinal_vento,
        theta_imp_rad, sinal_imperfeicao,
    )
    primeira = resolver_sistema(K_E, F, params, fundacao)

    normais = normais_elementos(params, acoes["reacoes"], acoes["peso_pilar_d_kN_m"])
    K_G = montar_matriz_geometrica(params, dofs_map, normais)
    K_2 = K_E - K_G
    livres, _ = graus_livres(params, fundacao)
    estabilidade_2 = verificar_positividade(K_2[np.ix_(livres, livres)])

    segunda = None
    erro_segunda = None
    if estabilidade_2["positiva_definida"]:
        try:
            K2ll = K_2[np.ix_(livres, livres)]
            d2_livres = np.linalg.solve(K2ll, F[livres])
            d2 = np.zeros(K_2.shape[0], dtype=float)
            d2[livres] = d2_livres
            segunda = {
                "d": d2,
                "reacoes": K_2 @ d2 - F,
                "diagnostico": estabilidade_2,
            }
        except np.linalg.LinAlgError as exc:
            erro_segunda = str(exc)
    else:
        erro_segunda = (
            "Perda de positividade de K_E-K_G: "
            f"menor autovalor = {estabilidade_2['lambda_min']:.4e}. "
            "Não foi calculado deslocamento de segunda ordem."
        )

    # Diagramas são pós-processados sob demanda para que a varredura de muitos cenários
    # não gaste tempo calculando curvas que talvez não sejam exibidas.
    elementos_1 = None
    elementos_2 = None
    if calcular_diagramas:
        elementos_1 = recuperar_elementos(
            params, primeira["d"], k_local, dofs_map, cargas_eq, q_imp,
            normais, incluir_geometria=False,
        )
        if segunda is not None:
            elementos_2 = recuperar_elementos(
                params, segunda["d"], k_local, dofs_map, cargas_eq, q_imp,
                normais, incluir_geometria=True,
            )

    return {
        "K_E": K_E,
        "K_G": K_G,
        "K_2": K_2,
        "F": F,
        "primeira_ordem": primeira,
        "segunda_ordem": segunda,
        "erro_segunda_ordem": erro_segunda,
        "estabilidade_segunda_ordem": estabilidade_2,
        "normais": normais,
        "elementos_1ordem": elementos_1,
        "elementos_2ordem": elementos_2,
        "vento_d": fator_vento * vento_k * sinal_vento,
        "sinal_vento": sinal_vento,
        "sinal_imperfeicao": sinal_imperfeicao,
        "cargas_eq": cargas_eq,
        "q_imp": q_imp,
        "dofs_map": dofs_map,
        "k_local": k_local,
    }


# =============================================================================
# 5. RECUPERAÇÃO DE ESFORÇOS E DIAGRAMAS
# =============================================================================

def integral_N_dv_array(
    d_local: np.ndarray,
    x_final: np.ndarray,
    L: float,
    n_base: float,
    n_topo: float,
    n_gauss: int = 5,
) -> np.ndarray:
    """Integra N(x)*v'(x) de 0 a cada x_final usando quadratura vetorizada."""
    xs = np.asarray(x_final, dtype=float)
    pontos, pesos = np.polynomial.legendre.leggauss(n_gauss)
    s = 0.5 * xs[:, None] * (pontos[None, :] + 1.0)
    w = 0.5 * xs[:, None] * pesos[None, :]
    n_s = n_base + (n_topo - n_base) * (s / L)
    dv_s = inclinacao_elemento(d_local, s, L)
    return np.sum(w * n_s * dv_s, axis=1)


def recuperar_elementos(
    params: ParametrosModelo,
    d_global: np.ndarray,
    k_local: np.ndarray,
    dofs_map: dict[tuple[int, int], list[int]],
    cargas_eq: dict[tuple[int, int], np.ndarray],
    q_imp: dict[tuple[int, int], float],
    normais: dict[tuple[int, int], dict[str, float]],
    incluir_geometria: bool,
    n_pontos: int = 41,
) -> dict[tuple[int, int], dict[str, Any]]:
    """Recupera esforços de extremidade e diagramas pelo equilíbrio do elemento.

    A parcela elástica das forças de extremidade fornece os momentos de seção nas pontas.
    Na segunda ordem, a variação de momento ao longo do elemento inclui a integral N(x)v'(x).
    A interpolação usa N(x) linear entre N_base e N_topo (peso próprio distribuído).
    """
    elementos: dict[tuple[int, int], dict[str, Any]] = {}
    L = params.h_pavimento
    x = np.linspace(0.0, L, n_pontos)
    for chave, dofs in dofs_map.items():
        dloc = d_global[dofs]
        f_eq = cargas_eq[chave]
        q = q_imp[chave]
        r_el = k_local @ dloc - f_eq
        # Convenção do diagrama: momento positivo conforme a convenção local adotada
        # para os esforços nodais [V_i, M_i, V_j, M_j].
        m_base = float(r_el[1])
        m_topo = -float(r_el[3])
        n_base = normais[chave]["N_base_kN"]
        n_topo = normais[chave]["N_topo_kN"]

        integrais = (
            integral_N_dv_array(dloc, x, L, n_base, n_topo)
            if incluir_geometria else np.zeros_like(x)
        )
        integral_total = float(integrais[-1])
        # M(x) = M(0) + V(0)x - q*x²/2 + integral_0^x[N(s)v'(s)ds].
        # V(0) é determinado para reproduzir também o momento de extremidade superior.
        q0 = (m_topo - m_base + 0.5 * q * L**2 - integral_total) / L
        momento = m_base + q0 * x - 0.5 * q * x**2 + integrais

        elementos[chave] = {
            "d_local": dloc.copy(),
            "forcas_extremidade_elasticas": r_el.copy(),
            "M_base_kNm": m_base,
            "M_topo_kNm": m_topo,
            "x_m": x.copy(),
            "M_kNm": momento,
            "q_horizontal_imperfeicao_kN_m": q,
            "N_base_kN": n_base,
            "N_topo_kN": n_topo,
            "integral_N_dv_kNm": integral_total,
            "Q0_reconstruido_kN": q0,
        }
    return elementos


# =============================================================================
# 6. GAMMA-Z — EXCLUSIVAMENTE COM DESLOCAMENTOS DE 1ª ORDEM
# =============================================================================

def integrar_deslocamento_vertical_elemento(d_local: np.ndarray, L: float) -> float:
    """Integra v(z) do elemento para o peso próprio distribuído."""
    pontos, pesos = np.polynomial.legendre.leggauss(5)
    x = 0.5 * L * (pontos + 1.0)
    w = 0.5 * L * pesos
    v = deslocamento_elemento(d_local, x, L)
    return float(np.sum(w * v))


def calcular_gamma_z(
    params: ParametrosModelo,
    resultado: dict[str, Any],
    acoes: dict[str, Any],
    vento_k: np.ndarray,
    fator_vento: float,
    theta_imp_rad: float,
) -> dict[str, Any]:
    """Calcula M1 e DeltaM usando somente o vetor de deslocamentos de 1ª ordem."""
    d1 = resultado["primeira_ordem"]["d"]
    sinal_vento = resultado["sinal_vento"]
    sinal_imp = resultado["sinal_imperfeicao"]
    z = params.h_pavimento * np.arange(1, params.n_pav + 1, dtype=float)
    vento_d = fator_vento * vento_k * sinal_vento

    m1_vento = float(np.sum(vento_d * z))
    m1_imp_vigas = 0.0
    for nivel in range(1, params.n_pav + 1):
        p_vigas = float(np.sum(acoes["reacoes"][nivel, :]))
        h_imp = sinal_imp * theta_imp_rad * p_vigas
        m1_imp_vigas += h_imp * nivel * params.h_pavimento

    peso_pilar_d = float(acoes["peso_pilar_d_kN_m"])
    m1_imp_pilares = 0.0
    for trecho in range(params.n_pav):
        z_centro = (trecho + 0.5) * params.h_pavimento
        q_h = sinal_imp * theta_imp_rad * peso_pilar_d
        m1_imp_pilares += N_PILARES * q_h * params.h_pavimento * z_centro

    m1_assinado = m1_vento + m1_imp_vigas + m1_imp_pilares

    # DeltaM das reações das vigas aplicadas nos níveis.
    delta_vigas = 0.0
    for nivel in range(1, params.n_pav + 1):
        p_nivel = float(np.sum(acoes["reacoes"][nivel, :]))
        delta_vigas += p_nivel * d1[idx_u(nivel)]

    # DeltaM do peso próprio de cada pilar: integral w_d * u(z) dz.
    delta_pilares = 0.0
    for coluna in range(N_PILARES):
        for trecho in range(params.n_pav):
            dofs = dofs_elemento(coluna, trecho, params.n_pav)
            dloc = d1[dofs]
            delta_pilares += peso_pilar_d * integrar_deslocamento_vertical_elemento(
                dloc, params.h_pavimento
            )

    delta_assinado = delta_vigas + delta_pilares
    if abs(m1_assinado) <= 1e-10:
        return {
            "M1_kNm": 0.0,
            "DeltaM_kNm": float("nan"),
            "razao": float("nan"),
            "gamma_z": None,
            "status": "Não aplicável: momento global de primeira ordem nulo.",
            "M1_vento_kNm": m1_vento,
            "M1_imperfeicao_vigas_kNm": m1_imp_vigas,
            "M1_imperfeicao_pilares_kNm": m1_imp_pilares,
            "DeltaM_vigas_kNm": delta_vigas,
            "DeltaM_pilares_kNm": delta_pilares,
        }

    delta_destabilizante = float(np.sign(m1_assinado) * delta_assinado)
    razao = delta_destabilizante / abs(m1_assinado)
    denominador = 1.0 - razao
    if denominador <= 0.0:
        gamma = None
        status = "Não finito: ΔM/M1 ≥ 1; o modelo indica perda do domínio de estabilidade."
    elif razao < 0.0:
        gamma = 1.0 / denominador
        status = "A razão resultou estabilizante; conferir sinais e associação das ações."
    else:
        gamma = 1.0 / denominador
        status = "Calculado com deslocamentos de 1ª ordem; comparação didática (menos de 4 pavimentos)."

    return {
        "M1_kNm": abs(m1_assinado),
        "M1_assinado_kNm": m1_assinado,
        "DeltaM_kNm": delta_destabilizante,
        "DeltaM_assinado_kNm": delta_assinado,
        "razao": razao,
        "gamma_z": gamma,
        "status": status,
        "M1_vento_kNm": m1_vento,
        "M1_imperfeicao_vigas_kNm": m1_imp_vigas,
        "M1_imperfeicao_pilares_kNm": m1_imp_pilares,
        "DeltaM_vigas_kNm": delta_vigas,
        "DeltaM_pilares_kNm": delta_pilares,
    }


# =============================================================================
# 7. HOGESLAG E REFERÊNCIA INDEPENDENTE POR AUTOVALORES
# =============================================================================

def calcular_hogeslag(
    params: ParametrosModelo,
    fundacao: str,
    acoes: dict[str, Any],
    fator_le: float,
    h_f_manual_m: float | None = None,
) -> dict[str, Any]:
    """Aplica as expressões de Hogeslag do enunciado com equivalências explícitas.

    le = fator_le * H_total é uma hipótese de comprimento equivalente; fator_le=2
    representa a referência de balanço com base engastada, mas deve ser justificado.
    Para F2, Ff=Kf_total/Hf. Hf usa o centroide das ações verticais, salvo entrada manual.
    A soma das ações verticais inclui reações de vigas em seus níveis e peso próprio dos
    pilares distribuído (integrado por trecho), não uma carga única arbitrariamente no topo.
    """
    p_total = float(acoes["carga_vertical_total_kN"])
    h_total = params.altura_total
    le = fator_le * h_total
    h_eq = float(acoes["h_eq_centroide_m"])
    h_f = float(h_f_manual_m) if h_f_manual_m is not None else h_eq
    ei_eq = params.ei_hogeslag_total

    if p_total <= 0.0 or le <= 0.0 or ei_eq <= 0.0:
        return {"status": "Não aplicável: parâmetros não positivos."}
    if not np.isfinite(h_f) or h_f <= 0.0:
        return {"status": "Não aplicável: altura de referência Hf inválida."}

    f_e = pi**2 * ei_eq / le**2
    if fundacao == "F1":
        f_f = float("inf")
        f_ref = f_e
    else:
        kf_total = N_PILARES * K_ROTACIONAL_PILAR
        f_f = kf_total / h_f
        f_ref = 1.0 / (1.0 / f_e + 1.0 / f_f)

    beta = f_ref / p_total
    if beta > 1.0:
        gamma_h = beta / (beta - 1.0)
        status = "Calculado pelas expressões do enunciado; equivalências Hf e le devem ser justificadas."
    else:
        gamma_h = None
        status = "β ≤ 1: a carga de cálculo alcança/supera a força de referência do modelo simplificado."

    return {
        "status": status,
        "Pvd_total_kN": p_total,
        "EI_eq_kNm2": ei_eq,
        "H_total_m": h_total,
        "H_eq_centroide_m": h_eq,
        "Hf_m": h_f,
        "fator_le": fator_le,
        "le_m": le,
        "Fe_kN": f_e,
        "Ff_kN": f_f,
        "Fref_kN": f_ref,
        "beta": beta,
        "gamma_hogeslag": gamma_h,
    }


def calcular_flambagem_autovalor(
    params: ParametrosModelo,
    fundacao: str,
    resultado: dict[str, Any],
) -> dict[str, Any]:
    """Fator crítico elástico de flambagem: K_G phi = mu K_E phi; lambda_cr=1/mu_max.

    É uma análise modal complementar, NÃO é o procedimento de Hogeslag.
    """
    K = resultado["K_E"]
    KG = resultado["K_G"]
    livres, _ = graus_livres(params, fundacao)
    Ke = K[np.ix_(livres, livres)]
    Kge = KG[np.ix_(livres, livres)]
    try:
        L = np.linalg.cholesky(0.5 * (Ke + Ke.T))
        A = np.linalg.solve(L, Kge)
        C = np.linalg.solve(L, A.T).T
        C = 0.5 * (C + C.T)
        mus = np.linalg.eigvalsh(C)
        mu_max = float(np.max(mus))
        if mu_max <= 1e-14:
            return {"status": "Sem autovalor geométrico positivo significativo.", "lambda_cr": None}
        lambda_cr = 1.0 / mu_max
        p_total = float(resultado["acoes"]["carga_vertical_total_kN"])
        p_cr = lambda_cr * p_total
        gamma_modal = lambda_cr / (lambda_cr - 1.0) if lambda_cr > 1.0 else None
        status = "Fator de carga crítico elástico por autovalores; referência independente." if lambda_cr > 1.0 else "lambda_cr ≤ 1: carga atual atinge/supera o fator crítico do modelo."
        return {
            "status": status,
            "mu_max": mu_max,
            "lambda_cr": lambda_cr,
            "Pcr_equiv_kN": p_cr,
            "gamma_modal": gamma_modal,
            "autovalores_transformados": mus,
        }
    except np.linalg.LinAlgError as exc:
        return {"status": f"Não foi possível calcular autovalores: {exc}", "lambda_cr": None}


# =============================================================================
# 8. CENÁRIOS, SOBRECARGAS E RESUMOS
# =============================================================================

def rotulo_sobrecarga(mascara: np.ndarray) -> str:
    rotulos = []
    for i in range(mascara.shape[0]):
        vaos = [f"V{j + 1}" for j in range(N_VAOS) if mascara[i, j]]
        if vaos:
            rotulos.append(f"N{i + 1}:{','.join(vaos)}")
    return "; ".join(rotulos) if rotulos else "Sem Q"


def gerar_mascaras_q(n_pav: int) -> list[np.ndarray]:
    """Todas as 2^(3*n_pav) distribuições de Q nos vãos/pavimentos."""
    n = n_pav * N_VAOS
    return [np.array(v, dtype=bool).reshape(n_pav, N_VAOS) for v in product([False, True], repeat=n)]


def sinais_aplicaveis(fator_vento: float, theta: float) -> list[tuple[int, int]]:
    sinais_w = [1, -1] if fator_vento > 0.0 else [0]
    sinais_imp = [1, -1] if theta > 0.0 else [0]
    return list(product(sinais_w, sinais_imp))


def executar_caso(
    params: ParametrosModelo,
    fundacao: str,
    combinacao: str,
    tipo_vento: str,
    mascara_q: np.ndarray,
    sinal_vento: int,
    sinal_imperfeicao: int,
    theta_imp_rad: float,
    fator_le_hogeslag: float,
    h_f_manual_m: float | None = None,
    calcular_diagramas: bool = False,
    calcular_modal: bool = False,
) -> dict[str, Any]:
    fatores = COMBINACOES[combinacao]
    acoes = calcular_acoes_verticais(params, mascara_q, combinacao)
    vento_k = distribuicao_vento(params, tipo_vento)
    resultado = resolver_analises_estruturais(
        params=params,
        fundacao=fundacao,
        acoes=acoes,
        vento_k=vento_k,
        fator_vento=fatores["gamma_w"],
        sinal_vento=sinal_vento,
        theta_imp_rad=theta_imp_rad,
        sinal_imperfeicao=sinal_imperfeicao,
        calcular_diagramas=calcular_diagramas,
    )
    resultado["acoes"] = acoes
    gamma = calcular_gamma_z(params, resultado, acoes, vento_k, fatores["gamma_w"], theta_imp_rad)
    hog = calcular_hogeslag(params, fundacao, acoes, fator_le_hogeslag, h_f_manual_m)
    modal = (
        calcular_flambagem_autovalor(params, fundacao, resultado)
        if calcular_modal else {"status": "Calculado sob demanda no caso detalhado.", "lambda_cr": None}
    )

    return {
        "n_pav": params.n_pav,
        "fundacao": fundacao,
        "combinacao": combinacao,
        "vento": tipo_vento,
        "mascara_q": mascara_q.copy(),
        "rotulo_q": rotulo_sobrecarga(mascara_q),
        "sinal_vento": sinal_vento,
        "sinal_imperfeicao": sinal_imperfeicao,
        "params": params,
        "acoes": acoes,
        "resultado": resultado,
        "gamma": gamma,
        "hogeslag": hog,
        "modal": modal,
    }


def registro_resumo(caso: dict[str, Any]) -> dict[str, Any]:
    g = caso["gamma"]
    h = caso["hogeslag"]
    m = caso["modal"]
    r2 = caso["resultado"]["segunda_ordem"]
    estabilidade = caso["resultado"]["estabilidade_segunda_ordem"]
    return {
        "Estrutura": f"E{caso['n_pav']}",
        "Fundação": caso["fundacao"],
        "Combinação": caso["combinacao"],
        "Vento": caso["vento"],
        "Sinal vento": caso["sinal_vento"],
        "Sinal imperfeição": caso["sinal_imperfeicao"],
        "Sobrecarga": caso["rotulo_q"],
        "ΣPvd (kN)": caso["acoes"]["carga_vertical_total_kN"],
        "M1 (kN.m)": g.get("M1_kNm"),
        "ΔM 1ª ordem (kN.m)": g.get("DeltaM_kNm"),
        "ΔM/M1": g.get("razao"),
        "γz": g.get("gamma_z"),
        "β Hogeslag": h.get("beta"),
        "γ Hogeslag": h.get("gamma_hogeslag"),
        "λcr modal": m.get("lambda_cr"),
        "γ modal": m.get("gamma_modal"),
        "λmin(K_E-K_G)": estabilidade.get("lambda_min"),
        "2ª ordem estável": bool(r2 is not None),
        "Status γz": g.get("status"),
        "Status Hogeslag": h.get("status"),
    }


def executar_lote(
    parametros: Iterable[ParametrosModelo],
    fundacoes: Iterable[str],
    combinacoes: Iterable[str],
    tipo_vento: str,
    mascaras_por_n: dict[int, list[np.ndarray]],
    theta_imp_rad: float,
    fator_le_hogeslag: float,
    h_f_manual_m: float | None = None,
    callback_progresso: Any = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    casos: list[dict[str, Any]] = []
    erros: list[str] = []
    tarefas = []
    for params in parametros:
        for fundacao in fundacoes:
            for combinacao in combinacoes:
                fatores = COMBINACOES[combinacao]
                mascaras = mascaras_por_n[params.n_pav]
                # Se gamma_Q=0, as máscaras são repetidas fisicamente; manter apenas Sem Q.
                if fatores["gamma_q"] == 0.0:
                    mascaras = [np.zeros((params.n_pav, N_VAOS), dtype=bool)]
                for mascara in mascaras:
                    for sw, si in sinais_aplicaveis(fatores["gamma_w"], theta_imp_rad):
                        tarefas.append((params, fundacao, combinacao, mascara, sw, si))

    total = len(tarefas)
    for i, (params, fundacao, combinacao, mascara, sw, si) in enumerate(tarefas, start=1):
        try:
            casos.append(
                executar_caso(
                    params=params,
                    fundacao=fundacao,
                    combinacao=combinacao,
                    tipo_vento=tipo_vento,
                    mascara_q=mascara,
                    sinal_vento=sw,
                    sinal_imperfeicao=si,
                    theta_imp_rad=theta_imp_rad,
                    fator_le_hogeslag=fator_le_hogeslag,
                    h_f_manual_m=h_f_manual_m,
                    calcular_diagramas=False,
                    calcular_modal=(total <= 300),
                )
            )
        except (np.linalg.LinAlgError, ValueError, FloatingPointError, KeyError) as exc:
            erros.append(
                f"E{params.n_pav}/{fundacao}/{combinacao}/{rotulo_sobrecarga(mascara)} "
                f"(vento={sw}, imperfeição={si}): {exc}"
            )
        if callback_progresso is not None and (
            i == 1 or i == total or i % max(1, total // 100) == 0
        ):
            callback_progresso(i, total)
    return casos, erros


# =============================================================================
# 9. INTERFACE STREAMLIT
# =============================================================================

def executar_interface() -> None:
    if st is None:
        raise RuntimeError(
            "Streamlit não está instalado. Execute: py -3.13 -m pip install streamlit numpy pandas"
        )

    st.set_page_config(page_title="Estabilidade — Hogeslag e γz", layout="wide")
    st.title("Estabilidade global — estruturas pré-moldadas")
    st.caption(
        "Análises de 1ª ordem, 2ª ordem geométrica linearizada, γz, Hogeslag e flambagem modal. "
        "Ferramenta acadêmica baseada no enunciado fornecido."
    )
    st.warning(
        "Atenção: γz é usado aqui como comparação didática em estruturas com menos de quatro pavimentos. "
        "Os parâmetros equivalentes de Hogeslag e as reduções de rigidez precisam ser justificados "
        "com a referência adotada na disciplina."
    )

    st.sidebar.header("1. Escopo da análise")
    escopo = st.sidebar.radio(
        "Modo",
        ["Caso individual", "Comparar E1/E2/E3 e F1/F2"],
        help="A comparação automática executa as seis combinações estrutura/fundação com uma mesma máscara de Q por nível.",
    )

    if escopo == "Caso individual":
        estrutura = st.sidebar.selectbox("Estrutura", ["E1 — 1 pavimento", "E2 — 2 pavimentos", "E3 — 3 pavimentos"], index=2)
        n_pav_sel = int(estrutura[1])
        fundacao_sel = st.sidebar.radio("Fundação", ["F1", "F2"], horizontal=True)
        lista_modelos = [(n_pav_sel, fundacao_sel)]
    else:
        st.sidebar.info("Serão analisadas E1/E2/E3 com F1/F2. O carregamento manual será definido para até três níveis.")
        lista_modelos = [(n, f) for n in (1, 2, 3) for f in ("F1", "F2")]
        n_pav_sel = 3
        fundacao_sel = "F1/F2"

    st.sidebar.header("2. Geometria e materiais")
    lado = st.sidebar.number_input("Lado da seção quadrada do pilar (m)", min_value=0.20, max_value=1.50, value=0.40, step=0.05)
    ec_gpa = st.sidebar.number_input("Ec (GPa)", min_value=10.0, max_value=60.0, value=30.0, step=1.0)
    fator_ei_1 = st.sidebar.number_input(
        "Fator de rigidez EI — análises estruturais", min_value=0.05, max_value=1.0,
        value=1.0, step=0.05,
        help="1,0 utiliza Ec.I bruto. Reduções devem ser justificadas; não se presume EI/3 automaticamente.",
    )
    fator_ei_hog = st.sidebar.number_input(
        "Fator de rigidez EIeq — Hogeslag", min_value=0.05, max_value=1.0,
        value=1.0, step=0.05,
        help="Aplicado à rigidez equivalente de quatro pilares em paralelo. Justificar a redução adotada.",
    )

    st.sidebar.header("3. Ações")
    combinacao_opcao = st.sidebar.selectbox(
        "Combinação ELU", ["Todas", "C1", "C2", "C3", "C4"], index=0,
        format_func=lambda c: c if c == "Todas" else f"{c} — {COMBINACOES[c]['descricao']}",
    )
    tipo_vento = st.sidebar.radio("Distribuição do vento", ["V1", "V2"], horizontal=True)
    theta_mrad = st.sidebar.number_input(
        "Desaprumo global θ (mrad)", min_value=0.0, max_value=20.0, value=0.0, step=0.1,
        help="Informe conforme o critério normativo adotado. O programa não determina θ automaticamente.",
    )
    theta = theta_mrad / 1000.0

    st.sidebar.header("4. Parâmetros equivalentes de Hogeslag")
    fator_le = st.sidebar.number_input(
        "Relação le/Htotal", min_value=0.50, max_value=4.00, value=2.00, step=0.10,
        help="Valor inicial 2,0 corresponde à referência de balanço com base engastada. Deve ser fundamentado para cada modelo/fundação.",
    )
    usar_hf_manual = st.sidebar.checkbox("Informar Hf manualmente para Ff (F2)", value=False)
    hf_manual = None
    if usar_hf_manual:
        hf_manual = st.sidebar.number_input("Hf (m)", min_value=0.10, max_value=100.0, value=6.0, step=0.5)

    st.markdown("### Distribuição da sobrecarga Qk")
    modo_q = st.radio(
        "Como definir Q?",
        ["Manual por vão e pavimento", "Varredura de todas as distribuições"],
        horizontal=True,
        disabled=(escopo != "Caso individual"),
    )
    mascara_3 = np.zeros((3, N_VAOS), dtype=bool)
    if modo_q == "Manual por vão e pavimento" or escopo != "Caso individual":
        n_linhas = 3 if escopo != "Caso individual" else n_pav_sel
        for nivel in range(n_linhas):
            with st.expander(f"Pavimento {nivel + 1} — cota {(nivel + 1) * H_PAVIMENTO_PADRAO:.0f} m", expanded=(nivel == 0)):
                cols = st.columns(3)
                for vao in range(N_VAOS):
                    mascara_3[nivel, vao] = cols[vao].checkbox(
                        f"Q no vão {vao + 1}", value=True,
                        key=f"q_{escopo}_{nivel}_{vao}",
                    )
    else:
        st.info("A varredura avaliará todas as 2^(3n) distribuições binárias de Q nos vãos e pavimentos.")

    combinacoes = list(COMBINACOES.keys()) if combinacao_opcao == "Todas" else [combinacao_opcao]
    executar = st.button("Executar análise", type="primary", use_container_width=True)

    if not executar:
        st.info("Configure as opções e clique em **Executar análise**.")
        return

    def criar_parametros(n: int) -> ParametrosModelo:
        return ParametrosModelo(
            n_pav=n,
            lado_pilar=float(lado),
            ec=float(ec_gpa) * 1e6,
            fator_ei_1ordem=float(fator_ei_1),
            fator_ei_hogeslag=float(fator_ei_hog),
        )

    if escopo == "Caso individual":
        params_list = [criar_parametros(n_pav_sel)]
        fundacoes = [fundacao_sel]
    else:
        # Uma parametrização por número de pavimentos; o lote fará o produto cartesiano
        # com F1/F2 sem duplicar cada estrutura.
        params_list = [criar_parametros(n) for n in (1, 2, 3)]
        fundacoes = ["F1", "F2"]

    if escopo == "Caso individual" and modo_q == "Varredura de todas as distribuições":
        mascaras_por_n = {n_pav_sel: gerar_mascaras_q(n_pav_sel)}
    else:
        mascaras_por_n = {}
        for n in {p.n_pav for p in params_list}:
            mascaras_por_n[n] = [mascara_3[:n, :].copy()]

    progresso = st.progress(0.0, text="Montando cenários...")
    casos, erros = executar_lote(
        parametros=params_list,
        fundacoes=fundacoes,
        combinacoes=combinacoes,
        tipo_vento=tipo_vento,
        mascaras_por_n=mascaras_por_n,
        theta_imp_rad=theta,
        fator_le_hogeslag=fator_le,
        h_f_manual_m=hf_manual,
        callback_progresso=lambda i, total: progresso.progress(
            i / max(total, 1), text=f"Analisando cenário {i}/{total}"
        ),
    )
    progresso.empty()

    if not casos:
        st.error("Nenhum cenário pôde ser resolvido. Verifique os parâmetros e os avisos abaixo.")
        for erro in erros[:30]:
            st.write(f"- {erro}")
        return

    df = pd.DataFrame([registro_resumo(c) for c in casos])
    st.success(f"Análise concluída: {len(casos)} cenários resolvidos; {len(erros)} cenários com erro.")
    if erros:
        with st.expander(f"Ver erros/instabilidades de {len(erros)} cenários"):
            for erro in erros[:100]:
                st.write(f"- {erro}")

    st.header("1. Resumo dos cenários")
    if len(casos) > 300:
        st.caption(
            "Para manter a varredura eficiente, o fator modal λcr é calculado sob demanda apenas "
            "para o caso detalhado; a triagem de estabilidade de K_E-K_G é feita em todos os cenários."
        )
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.download_button(
        "Baixar resumo dos cenários (CSV)",
        data=df.to_csv(index=False).encode("utf-8-sig"),
        file_name="resumo_estabilidade_hogeslag_gamma_z.csv",
        mime="text/csv",
    )

    # Comparação das seis configurações (ou resumo do caso selecionado).
    if escopo != "Caso individual":
        st.header("2. Comparação entre E1/E2/E3 e F1/F2")
        colunas_comp = [
            "Estrutura", "Fundação", "Combinação", "Sobrecarga", "ΣPvd (kN)",
            "γz", "γ Hogeslag", "λcr modal", "2ª ordem estável",
        ]
        st.dataframe(df[colunas_comp], use_container_width=True, hide_index=True)

    # Seleciona caso governante pela maior razão positiva ΔM/M1 válida.
    razoes = pd.to_numeric(df["ΔM/M1"], errors="coerce")
    validos = razoes[np.isfinite(razoes)]
    if len(validos) > 0:
        idx_critico = int(validos.idxmax())
    else:
        drift = []
        for c in casos:
            d = c["resultado"]["primeira_ordem"]["d"]
            drift.append(float(np.max(np.abs([d[idx_u(i)] for i in range(1, c["n_pav"] + 1)]))))
        idx_critico = int(np.argmax(drift))
    caso = casos[idx_critico]
    p = caso["params"]
    a = caso["acoes"]
    res = caso["resultado"]
    # Recupera os diagramas apenas para o caso detalhado. A varredura completa não
    # calcula diagramas em cada cenário até saber quais curvas serão exibidas.
    res["elementos_1ordem"] = recuperar_elementos(
        p, res["primeira_ordem"]["d"], res["k_local"], res["dofs_map"],
        res["cargas_eq"], res["q_imp"], res["normais"], incluir_geometria=False,
    )
    res["elementos_2ordem"] = None
    if res["segunda_ordem"] is not None:
        res["elementos_2ordem"] = recuperar_elementos(
            p, res["segunda_ordem"]["d"], res["k_local"], res["dofs_map"],
            res["cargas_eq"], res["q_imp"], res["normais"], incluir_geometria=True,
        )
    if caso["modal"].get("lambda_cr") is None:
        caso["modal"] = calcular_flambagem_autovalor(p, caso["fundacao"], res)

    st.header("3. Caso selecionado para detalhamento")
    st.caption(
        f"E{caso['n_pav']} | {caso['fundacao']} | {caso['combinacao']} | {caso['vento']} | "
        f"{caso['rotulo_q']} | sinal vento={caso['sinal_vento']} | sinal imperfeição={caso['sinal_imperfeicao']}"
    )

    gamma = caso["gamma"]
    hog = caso["hogeslag"]
    modal = caso["modal"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("M1,tot,d", f"{gamma.get('M1_kNm', float('nan')):.2f} kN.m" if np.isfinite(gamma.get("M1_kNm", np.nan)) else "Não aplicável")
    c2.metric("ΔM,tot,d (1ª ordem)", f"{gamma.get('DeltaM_kNm', float('nan')):.2f} kN.m" if np.isfinite(gamma.get("DeltaM_kNm", np.nan)) else "Não aplicável")
    c3.metric("γz didático", f"{gamma['gamma_z']:.4f}" if gamma.get("gamma_z") is not None else "Não definido")
    c4.metric("λcr modal", f"{modal['lambda_cr']:.4f}" if modal.get("lambda_cr") is not None else "Não calculado")
    st.info(gamma["status"])

    st.subheader("3.1 Hogeslag — expressões do enunciado")
    st.write(hog.get("status", "Parâmetros não disponíveis."))
    if "Fe_kN" in hog:
        h1, h2, h3, h4 = st.columns(4)
        h1.metric("Fe", f"{hog['Fe_kN']:.2f} kN")
        h2.metric("Ff", "∞" if not np.isfinite(hog["Ff_kN"]) else f"{hog['Ff_kN']:.2f} kN")
        h3.metric("Fref", f"{hog['Fref_kN']:.2f} kN")
        h4.metric("β", f"{hog['beta']:.4f}")
        st.write(
            f"EIeq = {hog['EI_eq_kNm2']:.2f} kN.m²; H_eq (centroide das ações) = "
            f"{hog['H_eq_centroide_m']:.3f} m; Hf = {hog['Hf_m']:.3f} m; "
            f"le = {hog['le_m']:.3f} m; ΣPvd = {hog['Pvd_total_kN']:.2f} kN."
        )
        st.write(
            f"Multiplicador γ_Hogeslag = **{hog['gamma_hogeslag']:.4f}**"
            if hog["gamma_hogeslag"] is not None else "Multiplicador γ_Hogeslag não finito pelo critério β ≤ 1."
        )
    st.caption(
        "H_eq é o centroide vertical das reações das vigas e do peso próprio distribuído dos pilares. "
        "O fator le/Htotal e Hf são hipóteses equivalentes explícitas; confirmar sua adequação com "
        "Hogeslag/El Debs antes de interpretar γ_Hogeslag como resultado validado."
    )

    st.subheader("3.2 Referência de flambagem elástica por autovalores")
    st.write(modal.get("status", "Não calculado."))
    if modal.get("lambda_cr") is not None:
        st.write(
            f"Fator de carga crítico λcr = **{modal['lambda_cr']:.4f}**; "
            f"Pcr equivalente = **{modal['Pcr_equiv_kN']:.2f} kN**."
        )
        st.caption("Esta análise é uma referência modal independente e não substitui o procedimento de Hogeslag.")

    st.header("4. Deslocamentos de primeira e segunda ordem")
    d1 = res["primeira_ordem"]["d"]
    d2 = res["segunda_ordem"]["d"] if res["segunda_ordem"] is not None else None
    linhas_d = []
    for nivel in range(p.n_pav + 1):
        linha = {
            "Nível": nivel,
            "Cota (m)": nivel * p.h_pavimento,
            "u 1ª ordem (mm)": d1[idx_u(nivel)] * 1000.0,
            "Deslocamento relativo 1ª ordem (mm)": (
                (d1[idx_u(nivel)] - d1[idx_u(nivel - 1)]) * 1000.0 if nivel > 0 else 0.0
            ),
        }
        if d2 is not None:
            linha["u 2ª ordem (mm)"] = d2[idx_u(nivel)] * 1000.0
            linha["Deslocamento relativo 2ª ordem (mm)"] = (
                (d2[idx_u(nivel)] - d2[idx_u(nivel - 1)]) * 1000.0 if nivel > 0 else 0.0
            )
        linhas_d.append(linha)
    df_d = pd.DataFrame(linhas_d)
    st.dataframe(df_d, use_container_width=True, hide_index=True)
    graf_cols = ["u 1ª ordem (mm)"] + (["u 2ª ordem (mm)"] if d2 is not None else [])
    st.line_chart(df_d.set_index("Cota (m)")[graf_cols])
    if d2 is None:
        st.error(res["erro_segunda_ordem"] or "A análise de segunda ordem não foi realizada.")

    st.header("5. Reações excêntricas e forças normais")
    linhas = []
    for nivel in range(1, p.n_pav + 1):
        for coluna in range(N_PILARES):
            linhas.append({
                "Nível": nivel,
                "Cota (m)": nivel * p.h_pavimento,
                "Pilar": coluna + 1,
                "Reação vertical das vigas (kN)": a["reacoes"][nivel, coluna],
                "Momento excêntrico (kN.m)": a["momentos_exc"][nivel, coluna],
            })
    st.dataframe(pd.DataFrame(linhas), use_container_width=True, hide_index=True)
    linhas_n = []
    for (coluna, trecho), ninfo in res["normais"].items():
        linhas_n.append({
            "Pilar": coluna + 1,
            "Trecho": f"{trecho + 1}-{trecho + 2}",
            **ninfo,
        })
    st.dataframe(pd.DataFrame(linhas_n), use_container_width=True, hide_index=True)

    st.header("6. Diagramas de momentos fletores")
    modo_diagrama = st.radio(
        "Ordem para os diagramas", ["Primeira ordem", "Segunda ordem linearizada"],
        horizontal=True,
        disabled=(res["segunda_ordem"] is None),
    )
    elems = res["elementos_1ordem"] if modo_diagrama == "Primeira ordem" or res["elementos_2ordem"] is None else res["elementos_2ordem"]
    for coluna in range(N_PILARES):
        with st.expander(f"Pilar {coluna + 1}", expanded=(coluna == 0)):
            series = {}
            for trecho in range(p.n_pav):
                item = elems[(coluna, trecho)]
                series[f"Trecho {trecho + 1}-{trecho + 2}"] = item["M_kNm"]
                st.write(
                    f"Trecho {trecho + 1}-{trecho + 2}: "
                    f"Mbase = {item['M_base_kNm']:.2f} kN.m; "
                    f"Mtopo = {item['M_topo_kNm']:.2f} kN.m"
                )
            df_m = pd.DataFrame(series, index=elems[(coluna, 0)]["x_m"])
            df_m.index.name = "Coordenada local (m)"
            st.line_chart(df_m)

    st.header("7. Envoltórias dos cenários analisados")
    st.caption(
        "As envoltórias são calculadas apenas para a mesma estrutura e fundação do caso detalhado, "
        "usando cenários efetivamente resolvidos e estáveis na segunda ordem. Para cobrir todas as "
        "distribuições parciais de Q, selecione a varredura completa no modo individual."
    )
    casos_estaveis = [
        c for c in casos
        if c["n_pav"] == p.n_pav
        and c["fundacao"] == caso["fundacao"]
        and c["resultado"]["segunda_ordem"] is not None
    ]
    if casos_estaveis:
        linhas_env = []
        for coluna in range(N_PILARES):
            for trecho in range(p.n_pav):
                chave = (coluna, trecho)
                curvas = []
                x = None
                for c_env in casos_estaveis:
                    r_env = c_env["resultado"]
                    if r_env["elementos_2ordem"] is None:
                        r_env["elementos_2ordem"] = recuperar_elementos(
                            c_env["params"], r_env["segunda_ordem"]["d"], r_env["k_local"],
                            r_env["dofs_map"], r_env["cargas_eq"], r_env["q_imp"],
                            r_env["normais"], incluir_geometria=True,
                        )
                    curvas.append(r_env["elementos_2ordem"][chave]["M_kNm"])
                    x = r_env["elementos_2ordem"][chave]["x_m"]
                pilhas = np.vstack(curvas)
                mmin = np.min(pilhas, axis=0)
                mmax = np.max(pilhas, axis=0)
                for xi, mn, mx in zip(x, mmin, mmax):
                    linhas_env.append({
                        "Pilar": coluna + 1,
                        "Trecho": f"{trecho + 1}-{trecho + 2}",
                        "x local (m)": xi,
                        "Mmin (kN.m)": mn,
                        "Mmax (kN.m)": mx,
                    })
                with st.expander(f"Envoltória — pilar {coluna + 1}, trecho {trecho + 1}-{trecho + 2}"):
                    st.line_chart(pd.DataFrame({"Mmin": mmin, "Mmax": mmax}, index=x))
        st.download_button(
            "Baixar envoltórias (CSV)",
            data=pd.DataFrame(linhas_env).to_csv(index=False).encode("utf-8-sig"),
            file_name="envoltorias_momentos_2a_ordem.csv",
            mime="text/csv",
        )
    else:
        st.warning("Nenhum cenário apresentou matriz de segunda ordem positiva definida; não há envoltória de 2ª ordem válida.")

    st.header("8. Reações de apoio e verificações")
    reacoes_globais = res["primeira_ordem"]["reacoes"]
    st.write(f"Reação horizontal total na base (1ª ordem): **{reacoes_globais[idx_u(0)]:.3f} kN**.")
    linhas_base = []
    for coluna in range(N_PILARES):
        idxb = idx_theta(coluna, 0, p.n_pav)
        elem_base = res["elementos_1ordem"][(coluna, 0)]
        linha = {
            "Pilar": coluna + 1,
            "Cortante na base do pilar (kN)": elem_base["forcas_extremidade_elasticas"][0],
            "Momento fletor no pé do pilar (kN.m)": elem_base["M_base_kNm"],
            "Força normal na base (kN)": res["normais"][(coluna, 0)]["N_base_kN"],
        }
        if caso["fundacao"] == "F1":
            linha["Reação generalizada de rotação na base (kN.m)"] = reacoes_globais[idxb]
        else:
            theta_base = d1[idxb]
            linha["Rotação da base (rad)"] = theta_base
            linha["Momento da mola (kN.m)"] = K_ROTACIONAL_PILAR * theta_base
        linhas_base.append(linha)
    st.dataframe(pd.DataFrame(linhas_base), use_container_width=True, hide_index=True)

    st.write(
        f"Menor autovalor de K_E-K_G: **{res['estabilidade_segunda_ordem']['lambda_min']:.4e}**; "
        f"número de condição da matriz elástica reduzida: **{res['primeira_ordem']['diagnostico']['condicao']:.3e}**."
    )
    st.caption(
        "A matriz K_E-K_G positiva definida é condição necessária para a solução linearizada usada. "
        "Autovalor nulo/negativo implica que os deslocamentos de 2ª ordem não devem ser apresentados como estáveis."
    )

    st.header("9. Verificações de equilíbrio e hipóteses")
    # Equilíbrio horizontal global: reação na base deve equilibrar a resultante externa.
    soma_h = float(np.sum(res["vento_d"]))
    for nivel in range(1, p.n_pav + 1):
        soma_h += (
            caso["sinal_imperfeicao"] * theta * float(np.sum(a["reacoes"][nivel, :]))
        )
    soma_h += (
        caso["sinal_imperfeicao"] * theta * a["peso_pilar_d_kN_m"]
        * N_PILARES * p.altura_total
    )
    residuo_h = float(reacoes_globais[idx_u(0)] + soma_h)
    st.write(f"Resíduo do equilíbrio horizontal global (1ª ordem): **{residuo_h:.6e} kN**.")
    soma_normais_base = sum(res["normais"][(c, 0)]["N_base_kN"] for c in range(N_PILARES))
    residuo_v = soma_normais_base - a["carga_vertical_total_kN"]
    st.write(f"Resíduo do equilíbrio vertical pela soma das normais de base: **{residuo_v:.6e} kN**.")

    with st.expander("Hipóteses e limitações para o relatório"):
        st.markdown(
            """
            1. Os quatro pilares têm translação horizontal compatibilizada em cada nível, enquanto suas rotações são independentes.
            2. As ligações viga-pilar são articuladas; as vigas transmitem reações verticais e impõem compatibilidade horizontal, sem rigidez rotacional.
            3. As reações e os momentos excêntricos são calculados vão a vão, preservando os sinais nos apoios adjacentes.
            4. O peso próprio dos pilares é distribuído ao longo da altura; a matriz geométrica integra a força normal linearmente variável ao longo de cada trecho.
            5. O cálculo de γz utiliza deslocamentos de primeira ordem e inclui as parcelas nodais e distribuídas do peso próprio dos pilares.
            6. A segunda ordem é linearizada por K_E-K_G, com forças normais de primeira ordem congeladas; não é análise materialmente não linear nem iteração geométrica completa.
            7. Hogeslag é calculado pelas expressões indicadas no enunciado. O comprimento equivalente e a altura de referência da fundação são hipóteses explícitas que precisam ser justificadas pela referência bibliográfica.
            8. A flambagem por autovalores é apresentada apenas como referência complementar independente, não como substituição de Hogeslag.
            9. γz é comparação didática para E1/E2/E3, pois o próprio enunciado alerta que as estruturas têm menos de quatro pavimentos.
            """
        )

    st.markdown("---")
    st.caption("Ferramenta acadêmica em desenvolvimento. Validar hipóteses, sinais, unidades e resultados com casos de referência antes de concluir a atividade.")


if __name__ == "__main__":
    executar_interface()
