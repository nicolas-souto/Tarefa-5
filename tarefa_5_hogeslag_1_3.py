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
py -3.13 -m pip install streamlit numpy pandas matplotlib reportlab
py -3.13 -m streamlit run tarefa_5_hogeslag_1_2_corrigido.py
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import importlib.util
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
    x, w, _, dN = _gauss_shapes(L, n_gauss)
    N_x = float(n_base) + (float(n_topo) - float(n_base)) * (x / L)
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


@lru_cache(maxsize=32)
def _gauss_legendre_cached(n: int) -> tuple[np.ndarray, np.ndarray]:
    pontos, pesos = np.polynomial.legendre.leggauss(int(n))
    pontos.setflags(write=False)
    pesos.setflags(write=False)
    return pontos, pesos


@lru_cache(maxsize=32)
def _gauss_shapes(L: float, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pontos, pesos = _gauss_legendre_cached(int(n))
    x = 0.5 * float(L) * (pontos + 1.0)
    w = 0.5 * float(L) * pesos
    N, dN = funcao_forma_hermite(x, float(L))
    for arr in (x, w, N, dN):
        arr.setflags(write=False)
    return x, w, N, dN


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


LIMITE_CONDICAO_QUASE_CRITICA = 1.0e5  # Alerta numérico, não limite normativo.


def verificar_positividade(K_reduzida: np.ndarray, tolerancia_relativa: float = 1e-10) -> dict[str, Any]:
    """Verifica a inércia após escalonamento diagonal e alerta sobre quase singularidade.

    O escalonamento D^-1 K D^-1, com Dii=sqrt(Kii), preserva a inércia da matriz
    por congruência e reduz o efeito da diferença de unidades entre translações e rotações.
    Os autovalores reportados são do sistema escalonado e, portanto, adimensionais.
    """
    Ksim = 0.5 * (K_reduzida + K_reduzida.T)
    autovalores_brutos = np.linalg.eigvalsh(Ksim)
    diagonal = np.diag(Ksim)
    if np.any(diagonal <= 0.0) or not np.all(np.isfinite(diagonal)):
        menor_bruto = float(np.min(autovalores_brutos))
        maior_bruto = float(np.max(autovalores_brutos))
        return {
            "positiva_definida": False,
            "lambda_min": menor_bruto,
            "lambda_max": maior_bruto,
            "lambda_min_bruto": menor_bruto,
            "lambda_max_bruto": maior_bruto,
            "condicao": float("inf"),
            "quase_critica": False,
            "autovalores": autovalores_brutos,
            "limite_tolerancia": tolerancia_relativa,
            "escala": "bruta (diagonal não positiva)",
        }
    escala = np.sqrt(diagonal)
    K_norm = Ksim / np.outer(escala, escala)
    autovalores = np.linalg.eigvalsh(0.5 * (K_norm + K_norm.T))
    menor = float(np.min(autovalores))
    maior = float(np.max(autovalores))
    limite = tolerancia_relativa * max(abs(maior), 1.0)
    positiva = menor > limite
    condicao = float(maior / menor) if positiva else float("inf")
    quase_critica = bool(positiva and condicao > LIMITE_CONDICAO_QUASE_CRITICA)
    return {
        "positiva_definida": positiva,
        "lambda_min": menor,
        "lambda_max": maior,
        "lambda_min_bruto": float(np.min(autovalores_brutos)),
        "lambda_max_bruto": float(np.max(autovalores_brutos)),
        "condicao": condicao,
        "quase_critica": quase_critica,
        "autovalores": autovalores,
        "limite_tolerancia": limite,
        "escala": "diagonal normalizada; autovalores adimensionais",
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
    pontos, pesos = _gauss_legendre_cached(n_gauss)
    s = 0.5 * xs[:, None] * (pontos[None, :] + 1.0)
    w = 0.5 * xs[:, None] * pesos[None, :]
    r = s / float(L)
    n_s = n_base + (n_topo - n_base) * r
    # Derivadas das funções de Hermite, avaliadas sem recriar pilhas de arrays.
    dv_s = (
        ((-6.0 * r + 6.0 * r**2) / L) * d_local[0]
        + (1.0 - 4.0 * r + 3.0 * r**2) * d_local[1]
        + ((6.0 * r - 6.0 * r**2) / L) * d_local[2]
        + (-2.0 * r + 3.0 * r**2) * d_local[3]
    )
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
    _, w, N, _ = _gauss_shapes(float(L), 5)
    v = np.sum(N * np.asarray(d_local, dtype=float), axis=1)
    return float(np.dot(w, v))


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
        "λmin normalizado(K_E-K_G)": estabilidade.get("lambda_min"),
        "Condicionamento normalizado K_E-K_G": estabilidade.get("condicao"),
        "2ª ordem quase crítica": estabilidade.get("quase_critica", False),
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
        f"Menor autovalor diagonal-normalizado de K_E-K_G: **{res['estabilidade_segunda_ordem']['lambda_min']:.4e}**; "
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



# =============================================================================
# 10. REVISÕES DO NÚCLEO E NOVOS MÓDULOS DE ANÁLISE
# =============================================================================

from dataclasses import replace as dataclass_replace
from datetime import datetime


def calcular_acoes_verticais(
    params: ParametrosModelo,
    mascara_q: np.ndarray,
    combinacao: str,
    gamma_g_override: float | None = None,
    gamma_q_override: float | None = None,
) -> dict[str, Any]:
    """Reações por vão, momentos excêntricos e cargas verticais de cálculo.

    gamma_g_override/gamma_q_override permitem executar uma verificação complementar
    com ações permanentes favoráveis. Essa verificação é separada das combinações C1-C4
    porque o enunciado não fornece o coeficiente favorável: o usuário deve informá-lo
    conforme a base normativa efetivamente adotada.
    """
    if mascara_q.shape != (params.n_pav, N_VAOS):
        raise ValueError("A máscara de Q deve ter dimensão (número de pavimentos, 3 vãos).")
    if combinacao not in COMBINACOES:
        raise ValueError(f"Combinação desconhecida: {combinacao}")

    fatores = COMBINACOES[combinacao]
    gamma_g = float(fatores["gamma_g"] if gamma_g_override is None else gamma_g_override)
    gamma_q = float(fatores["gamma_q"] if gamma_q_override is None else gamma_q_override)
    if gamma_g < 0.0 or gamma_q < 0.0:
        raise ValueError("Os fatores de ponderação não podem ser negativos.")

    reacoes = np.zeros((params.n_pav + 1, N_PILARES), dtype=float)
    momentos = np.zeros_like(reacoes)
    g_d = gamma_g * params.gk_viga
    q_d = gamma_q * params.qk_viga

    for nivel in range(1, params.n_pav + 1):
        for vao in range(N_VAOS):
            q_linear = g_d + q_d * float(mascara_q[nivel - 1, vao])
            reacao = q_linear * params.l_vao / 2.0
            p_esq, p_dir = vao, vao + 1
            reacoes[nivel, p_esq] += reacao
            reacoes[nivel, p_dir] += reacao
            momentos[nivel, p_esq] -= params.excentricidade * reacao
            momentos[nivel, p_dir] += params.excentricidade * reacao

    peso_pilar_d = gamma_g * params.peso_proprio_pilar
    peso_pilares_total = N_PILARES * params.n_pav * params.h_pavimento * peso_pilar_d
    carga_vigas_total = float(np.sum(reacoes))
    carga_total = carga_vigas_total + peso_pilares_total

    cargas_niveis = np.sum(reacoes[1:, :], axis=1)
    cotas = params.h_pavimento * np.arange(1, params.n_pav + 1, dtype=float)
    momento_estatico = float(np.sum(cargas_niveis * cotas))
    peso_por_trecho = N_PILARES * peso_pilar_d * params.h_pavimento
    z_centros = params.h_pavimento * (np.arange(params.n_pav, dtype=float) + 0.5)
    momento_estatico += float(np.sum(peso_por_trecho * z_centros))
    h_eq = momento_estatico / carga_total if carga_total > 0.0 else float("nan")

    return {
        "reacoes": reacoes,
        "momentos_exc": momentos,
        "peso_pilar_d_kN_m": peso_pilar_d,
        "carga_vigas_total_kN": carga_vigas_total,
        "peso_pilares_total_kN": peso_pilares_total,
        "carga_vertical_total_kN": carga_total,
        "cargas_vigas_nivel_kN": cargas_niveis,
        "h_eq_centroide_m": h_eq,
        "gamma_g_utilizado": gamma_g,
        "gamma_q_utilizado": gamma_q,
    }


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
    gamma_g_override: float | None = None,
    gamma_q_override: float | None = None,
    rotulo_cenario: str = "Combinação do enunciado",
) -> dict[str, Any]:
    fatores = COMBINACOES[combinacao]
    acoes = calcular_acoes_verticais(
        params, mascara_q, combinacao,
        gamma_g_override=gamma_g_override,
        gamma_q_override=gamma_q_override,
    )
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
        calcular_diagramas=False,
    )
    resultado["acoes"] = acoes
    gamma = calcular_gamma_z(params, resultado, acoes, vento_k, fatores["gamma_w"], theta_imp_rad)
    hog = calcular_hogeslag(params, fundacao, acoes, fator_le_hogeslag, h_f_manual_m)
    modal = (
        calcular_flambagem_autovalor(params, fundacao, resultado)
        if calcular_modal else {"status": "Calculado sob demanda.", "lambda_cr": None}
    )

    caso = {
        "n_pav": params.n_pav,
        "fundacao": fundacao,
        "combinacao": combinacao,
        "vento": tipo_vento if fatores["gamma_w"] > 0 else "Sem vento",
        "tipo_vento_modelo": tipo_vento,
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
        "rotulo_cenario": rotulo_cenario,
    }
    if calcular_diagramas:
        resultado["elementos_1ordem"] = recuperar_elementos_revisado(
            params, resultado["primeira_ordem"]["d"], resultado["K_E"],
            resultado["K_G"], resultado["k_local"], resultado["dofs_map"],
            resultado["cargas_eq"], resultado["q_imp"], resultado["normais"],
            incluir_geometria=False,
        )
        if resultado["segunda_ordem"] is not None:
            resultado["elementos_2ordem"] = recuperar_elementos_revisado(
                params, resultado["segunda_ordem"]["d"], resultado["K_E"],
                resultado["K_G"], resultado["k_local"], resultado["dofs_map"],
                resultado["cargas_eq"], resultado["q_imp"], resultado["normais"],
                incluir_geometria=True,
            )
        else:
            resultado["elementos_2ordem"] = None
    return caso


def calcular_gamma_z(
    params: ParametrosModelo,
    resultado: dict[str, Any],
    acoes: dict[str, Any],
    vento_k: np.ndarray,
    fator_vento: float,
    theta_imp_rad: float,
) -> dict[str, Any]:
    """Gamma-z didático calculado estritamente com deslocamentos de 1ª ordem."""
    d1 = resultado["primeira_ordem"]["d"]
    sinal_vento = int(resultado["sinal_vento"])
    sinal_imp = int(resultado["sinal_imperfeicao"])
    cotas = params.h_pavimento * np.arange(1, params.n_pav + 1, dtype=float)
    vento_d = fator_vento * vento_k * sinal_vento

    m1_vento = float(np.sum(vento_d * cotas))
    m1_imp_vigas = 0.0
    for nivel in range(1, params.n_pav + 1):
        p_nivel = float(np.sum(acoes["reacoes"][nivel, :]))
        h_imp = sinal_imp * theta_imp_rad * p_nivel
        m1_imp_vigas += h_imp * cotas[nivel - 1]

    peso_pilar_d = float(acoes["peso_pilar_d_kN_m"])
    q_imp = sinal_imp * theta_imp_rad * peso_pilar_d
    m1_imp_pilares = sum(
        N_PILARES * q_imp * params.h_pavimento * (i + 0.5) * params.h_pavimento
        for i in range(params.n_pav)
    )
    m1_assinado = m1_vento + m1_imp_vigas + m1_imp_pilares

    delta_vigas = sum(
        float(np.sum(acoes["reacoes"][nivel, :])) * d1[idx_u(nivel)]
        for nivel in range(1, params.n_pav + 1)
    )
    delta_pilares = 0.0
    for coluna in range(N_PILARES):
        for trecho in range(params.n_pav):
            dofs = dofs_elemento(coluna, trecho, params.n_pav)
            delta_pilares += peso_pilar_d * integrar_deslocamento_vertical_elemento(
                d1[dofs], params.h_pavimento
            )
    delta_assinado = float(delta_vigas + delta_pilares)

    base = {
        "M1_vento_kNm": m1_vento,
        "M1_imperfeicao_vigas_kNm": float(m1_imp_vigas),
        "M1_imperfeicao_pilares_kNm": float(m1_imp_pilares),
        "DeltaM_vigas_kNm": float(delta_vigas),
        "DeltaM_pilares_kNm": float(delta_pilares),
        "M1_assinado_kNm": float(m1_assinado),
        "DeltaM_assinado_kNm": delta_assinado,
    }
    if abs(m1_assinado) <= 1e-10:
        return {
            **base,
            "M1_kNm": 0.0,
            "DeltaM_kNm": float("nan"),
            "razao": float("nan"),
            "gamma_z": None,
            "status": "Não aplicável: momento global de 1ª ordem nulo ou praticamente nulo.",
        }

    delta_proj = float(np.sign(m1_assinado) * delta_assinado)
    razao = delta_proj / abs(m1_assinado)
    if delta_proj < -1e-10:
        return {
            **base,
            "M1_kNm": abs(m1_assinado),
            "DeltaM_kNm": delta_proj,
            "razao": razao,
            "gamma_z": None,
            "status": "Não interpretável como γz: ΔM projetado é estabilizante. Conferir combinação/sinais.",
        }
    if razao >= 1.0:
        return {
            **base,
            "M1_kNm": abs(m1_assinado),
            "DeltaM_kNm": delta_proj,
            "razao": razao,
            "gamma_z": None,
            "status": "Não finito: ΔM/M1 ≥ 1; o procedimento não fornece γz finito.",
        }
    if razao < 0.0:
        return {
            **base,
            "M1_kNm": abs(m1_assinado),
            "DeltaM_kNm": delta_proj,
            "razao": razao,
            "gamma_z": None,
            "status": "Não aplicável: razão negativa; revisar os sinais das ações e deslocamentos.",
        }
    gamma = 1.0 / (1.0 - razao)
    return {
        **base,
        "M1_kNm": abs(m1_assinado),
        "DeltaM_kNm": delta_proj,
        "razao": razao,
        "gamma_z": gamma,
        "status": "Calculado com deslocamentos de 1ª ordem; uso comparativo didático para E1/E2/E3.",
    }


def recuperar_elementos_revisado(
    params: ParametrosModelo,
    d_global: np.ndarray,
    K_E: np.ndarray,
    K_G: np.ndarray,
    k_local: np.ndarray,
    dofs_map: dict[tuple[int, int], list[int]],
    cargas_eq: dict[tuple[int, int], np.ndarray],
    q_imp: dict[tuple[int, int], float],
    normais: dict[tuple[int, int], dict[str, float]],
    incluir_geometria: bool,
    n_pontos: int = 61,
) -> dict[tuple[int, int], dict[str, Any]]:
    """Recupera diagramas pelo equilíbrio do elemento e registra resíduos de extremidade.

    Na análise de 2ª ordem, o cortante inicial é obtido do vetor de extremidade
    (K_E-K_G)d-f; a parcela integral N(x)*v'(x) entra no equilíbrio do momento.
    Os momentos de extremidade de flexão são recuperados de K_E d-f. A diferença
    no momento do topo é exposta como indicador de consistência da recuperação.
    """
    saida: dict[tuple[int, int], dict[str, Any]] = {}
    L = params.h_pavimento
    x = np.linspace(0.0, L, n_pontos)
    for chave, dofs in dofs_map.items():
        dloc = d_global[dofs]
        f_eq = cargas_eq[chave]
        q = float(q_imp[chave])
        r_el = k_local @ dloc - f_eq
        m_base = float(r_el[1])
        m_topo = -float(r_el[3])
        n_base = float(normais[chave]["N_base_kN"])
        n_topo = float(normais[chave]["N_topo_kN"])

        if incluir_geometria:
            kg_local = matriz_rigidez_geometrica_variavel(n_base, n_topo, L)
            r_total = (k_local - kg_local) @ dloc - f_eq
            cortante_inicial = -float(r_total[0])
            integral = integral_N_dv_array(dloc, x, L, n_base, n_topo)
        else:
            r_total = r_el.copy()
            cortante_inicial = -float(r_el[0])
            integral = np.zeros_like(x)

        momento = m_base + cortante_inicial * x - 0.5 * q * x**2 + integral
        m_topo_reconstruido = float(momento[-1])
        residuo_topo = m_topo_reconstruido - m_topo
        escala = max(abs(m_base), abs(m_topo), 1.0)

        saida[chave] = {
            "d_local": dloc.copy(),
            "forcas_extremidade_elasticas": r_el.copy(),
            "forcas_extremidade_totais": r_total.copy(),
            "M_base_kNm": m_base,
            "M_topo_kNm": m_topo,
            "M_topo_reconstruido_kNm": m_topo_reconstruido,
            "residuo_momento_topo_kNm": residuo_topo,
            "residuo_momento_topo_relativo": abs(residuo_topo) / escala,
            "x_m": x.copy(),
            "M_kNm": momento,
            "q_horizontal_imperfeicao_kN_m": q,
            "N_base_kN": n_base,
            "N_topo_kN": n_topo,
            "integral_N_dv_kNm": float(integral[-1]),
            "cortante_inicial_equilibrio_kN": cortante_inicial,
            "incluir_geometria": incluir_geometria,
        }
    return saida


def rotulo_caso(meta: dict[str, Any]) -> str:
    qtxt = rotulo_sobrecarga(meta["mascara_q"])
    vento = meta["tipo_vento"] if meta["fator_vento"] > 0 else "Sem vento"
    sw = "+" if meta["sinal_vento"] > 0 else ("−" if meta["sinal_vento"] < 0 else "0")
    si = "+" if meta["sinal_imperfeicao"] > 0 else ("−" if meta["sinal_imperfeicao"] < 0 else "0")
    extra = f" | {meta['rotulo_cenario']}" if meta.get("gamma_g_override") is not None else ""
    return (
        f"E{meta['n_pav']}/{meta['fundacao']}/{meta['combinacao']}/{vento}/"
        f"W{sw}/I{si}/{qtxt}{extra}"
    )


def criar_meta_caso(
    params: ParametrosModelo,
    fundacao: str,
    combinacao: str,
    tipo_vento: str,
    mascara_q: np.ndarray,
    sinal_vento: int,
    sinal_imperfeicao: int,
    theta: float,
    fator_le: float,
    h_f_manual: float | None,
    gamma_g_override: float | None = None,
    gamma_q_override: float | None = None,
    rotulo_cenario: str = "Combinação do enunciado",
) -> dict[str, Any]:
    return {
        "params": params,
        "n_pav": params.n_pav,
        "fundacao": fundacao,
        "combinacao": combinacao,
        "tipo_vento": tipo_vento,
        "mascara_q": mascara_q.copy(),
        "sinal_vento": sinal_vento,
        "sinal_imperfeicao": sinal_imperfeicao,
        "theta": theta,
        "fator_le": fator_le,
        "h_f_manual": h_f_manual,
        "gamma_g_override": gamma_g_override,
        "gamma_q_override": gamma_q_override,
        "rotulo_cenario": rotulo_cenario,
        "fator_vento": COMBINACOES[combinacao]["gamma_w"],
    }


def executar_meta(meta: dict[str, Any], calcular_diagramas: bool = True, calcular_modal: bool = True) -> dict[str, Any]:
    return executar_caso(
        params=meta["params"],
        fundacao=meta["fundacao"],
        combinacao=meta["combinacao"],
        tipo_vento=meta["tipo_vento"],
        mascara_q=meta["mascara_q"],
        sinal_vento=meta["sinal_vento"],
        sinal_imperfeicao=meta["sinal_imperfeicao"],
        theta_imp_rad=meta["theta"],
        fator_le_hogeslag=meta["fator_le"],
        h_f_manual_m=meta["h_f_manual"],
        calcular_diagramas=calcular_diagramas,
        calcular_modal=calcular_modal,
        gamma_g_override=meta["gamma_g_override"],
        gamma_q_override=meta["gamma_q_override"],
        rotulo_cenario=meta["rotulo_cenario"],
    )


def atualizar_envoltoria(
    acumulador: dict[str, Any],
    elementos: dict[tuple[int, int], dict[str, Any]],
    case_id: str,
) -> None:
    for chave, item in elementos.items():
        curva = np.asarray(item["M_kNm"], dtype=float)
        atual_min = acumulador["min"][chave]
        atual_max = acumulador["max"][chave]
        mask_min = curva < atual_min
        mask_max = curva > atual_max
        atual_min[mask_min] = curva[mask_min]
        atual_max[mask_max] = curva[mask_max]
        acumulador["id_min"][chave][mask_min] = case_id
        acumulador["id_max"][chave][mask_max] = case_id


def inicializar_envoltoria(params: ParametrosModelo, n_pontos: int = 31) -> dict[str, Any]:
    x = np.linspace(0.0, params.h_pavimento, n_pontos)
    chaves = [(c, t) for c in range(N_PILARES) for t in range(params.n_pav)]
    return {
        "x": x,
        "min": {k: np.full(n_pontos, np.inf) for k in chaves},
        "max": {k: np.full(n_pontos, -np.inf) for k in chaves},
        "id_min": {k: np.full(n_pontos, "", dtype=object) for k in chaves},
        "id_max": {k: np.full(n_pontos, "", dtype=object) for k in chaves},
        "n_casos": 0,
    }


def atualizar_governador(
    governadores: dict[tuple[str, int, str], dict[str, Any]],
    chave: tuple[str, int, str],
    valor: float,
    meta: dict[str, Any],
    sentido: str,
) -> None:
    try:
        valor_num = float(valor)
    except (TypeError, ValueError):
        return
    if not np.isfinite(valor_num):
        return
    valor = valor_num
    atual = governadores.get(chave)
    melhor = atual is None or (valor > atual["valor"] if sentido == "max" else valor < atual["valor"])
    if melhor:
        governadores[chave] = {"valor": float(valor), "meta": meta.copy(), "rotulo": rotulo_caso(meta)}


def executar_lote_modular(
    parametros: list[ParametrosModelo],
    fundacoes: list[str],
    combinacoes: list[str],
    tipos_vento: list[str],
    mascaras_por_n: dict[int, list[np.ndarray]],
    theta_imp_rad: float,
    fator_le_hogeslag: float,
    h_f_manual_m: float | None = None,
    incluir_g_favoravel: bool = False,
    gamma_g_favoravel: float = 1.0,
    callback_progresso: Any = None,
) -> tuple[pd.DataFrame, dict[tuple[int, str, str], dict[str, Any]], dict[tuple[str, int, str], dict[str, Any]], dict[str, dict[str, Any]], list[str]]:
    """Varre casos e atualiza envoltórias sem armazenar matrizes de todos os cenários."""
    tarefas: list[dict[str, Any]] = []
    for params in parametros:
        for fundacao in fundacoes:
            for combinacao in combinacoes:
                fatores = COMBINACOES[combinacao]
                mascaras = mascaras_por_n[params.n_pav]
                if fatores["gamma_q"] == 0.0:
                    mascaras = [np.zeros((params.n_pav, N_VAOS), dtype=bool)]
                winds = tipos_vento if fatores["gamma_w"] > 0.0 else [tipos_vento[0] if tipos_vento else "V1"]
                for tipo_vento in winds:
                    sinais = sinais_aplicaveis(fatores["gamma_w"], theta_imp_rad)
                    for mascara in mascaras:
                        for sw, si in sinais:
                            tarefas.append(criar_meta_caso(
                                params, fundacao, combinacao, tipo_vento, mascara, sw, si,
                                theta_imp_rad, fator_le_hogeslag, h_f_manual_m,
                            ))

            if incluir_g_favoravel:
                mascara_zero = np.zeros((params.n_pav, N_VAOS), dtype=bool)
                for tipo_vento in tipos_vento:
                    for sw, si in sinais_aplicaveis(COMBINACOES["C4"]["gamma_w"], theta_imp_rad):
                        tarefas.append(criar_meta_caso(
                            params, fundacao, "C4", tipo_vento, mascara_zero, sw, si,
                            theta_imp_rad, fator_le_hogeslag, h_f_manual_m,
                            gamma_g_override=gamma_g_favoravel,
                            gamma_q_override=0.0,
                            rotulo_cenario="G favorável — verificação complementar",
                        ))

    total = len(tarefas)
    registros: list[dict[str, Any]] = []
    metas: dict[str, dict[str, Any]] = {}
    governadores: dict[tuple[str, int, str], dict[str, Any]] = {}
    envelopes: dict[tuple[str, int, str], dict[str, Any]] = {}
    erros: list[str] = []

    for i, meta in enumerate(tarefas, start=1):
        case_id = f"CASO-{i:06d}"
        try:
            caso = executar_meta(meta, calcular_diagramas=False, calcular_modal=False)
            res = caso["resultado"]
            d1 = res["primeira_ordem"]["d"]
            drifts = [abs(d1[idx_u(n)] - d1[idx_u(n - 1)]) for n in range(1, meta["n_pav"] + 1)]
            drift_max_mm = max(drifts, default=0.0) * 1000.0
            n_base_max = max(res["normais"][(c, 0)]["N_base_kN"] for c in range(N_PILARES))
            lam_min = float(res["estabilidade_segunda_ordem"]["lambda_min"])
            beta = caso["hogeslag"].get("beta", np.nan)
            gamma_z = caso["gamma"].get("gamma_z")

            row = registro_resumo(caso)
            row.update({
                "ID": case_id,
                "Cenário adicional": meta["rotulo_cenario"],
                "Fator γG": caso["acoes"]["gamma_g_utilizado"],
                "Fator γQ": caso["acoes"]["gamma_q_utilizado"],
                "Maior deslocamento relativo 1ª ordem (mm)": drift_max_mm,
                "Maior normal na base (kN)": n_base_max,
                "Condicionamento normalizado K_E-K_G": res["estabilidade_segunda_ordem"]["condicao"],
                "2ª ordem quase crítica": res["estabilidade_segunda_ordem"].get("quase_critica", False),
            })
            registros.append(row)
            metas[case_id] = meta

            for fund_metric, value, sense in [
                ("gamma_z_max", gamma_z, "max"),
                ("razao_delta_m1_max", caso["gamma"].get("razao"), "max"),
                ("drift_1ordem_max_mm", drift_max_mm, "max"),
                ("N_base_max_kN", n_base_max, "max"),
                ("beta_hogeslag_min", beta, "min"),
                ("lambda_min_KE_KG", lam_min, "min"),
            ]:
                atualizar_governador(governadores, (fund_metric, meta["n_pav"], meta["fundacao"]), value, meta, sense)

            # Envoltórias de primeira ordem para todos os casos resolvidos.
            env_key_1 = ("1ª ordem", meta["n_pav"], meta["fundacao"])
            if env_key_1 not in envelopes:
                envelopes[env_key_1] = inicializar_envoltoria(meta["params"])
            elems1 = recuperar_elementos_revisado(
                meta["params"], res["primeira_ordem"]["d"], res["K_E"], res["K_G"],
                res["k_local"], res["dofs_map"], res["cargas_eq"], res["q_imp"],
                res["normais"], incluir_geometria=False, n_pontos=31,
            )
            atualizar_envoltoria(envelopes[env_key_1], elems1, case_id)
            envelopes[env_key_1]["n_casos"] += 1

            # Envoltórias de 2ª ordem são separadas por qualidade numérica.
            # Casos quase críticos NUNCA entram na envoltória regular.
            if res["segunda_ordem"] is not None:
                quase = bool(res["estabilidade_segunda_ordem"].get("quase_critica", False))
                ordem_env = "2ª ordem quase crítica" if quase else "2ª ordem linearizada"
                env_key_2 = (ordem_env, meta["n_pav"], meta["fundacao"])
                if env_key_2 not in envelopes:
                    envelopes[env_key_2] = inicializar_envoltoria(meta["params"])
                elems2 = recuperar_elementos_revisado(
                    meta["params"], res["segunda_ordem"]["d"], res["K_E"], res["K_G"],
                    res["k_local"], res["dofs_map"], res["cargas_eq"], res["q_imp"],
                    res["normais"], incluir_geometria=True, n_pontos=31,
                )
                atualizar_envoltoria(envelopes[env_key_2], elems2, case_id)
                envelopes[env_key_2]["n_casos"] += 1
        except (np.linalg.LinAlgError, ValueError, FloatingPointError, KeyError) as exc:
            erros.append(f"{rotulo_caso(meta)}: {exc}")
        if callback_progresso is not None and (i == total or i == 1 or i % max(1, total // 100) == 0):
            callback_progresso(i, total)

    # Governantes globais de momentos são extraídos diretamente das envoltórias.
    for (ordem, n, fund), env in envelopes.items():
        if not env["min"]:
            continue
        maior = (-np.inf, None, None)
        menor = (np.inf, None, None)
        for chave, curva in env["max"].items():
            j = int(np.argmax(curva))
            if curva[j] > maior[0]:
                maior = (float(curva[j]), chave, str(env["id_max"][chave][j]))
        for chave, curva in env["min"].items():
            j = int(np.argmin(curva))
            if curva[j] < menor[0]:
                menor = (float(curva[j]), chave, str(env["id_min"][chave][j]))
        if maior[1] is not None and maior[2] in metas:
            governadores[(f"Mmax_{ordem}", n, fund)] = {
                "valor": maior[0], "elemento": maior[1], "id": maior[2],
                "meta": metas[maior[2]], "rotulo": rotulo_caso(metas[maior[2]]),
            }
        if menor[1] is not None and menor[2] in metas:
            governadores[(f"Mmin_{ordem}", n, fund)] = {
                "valor": menor[0], "elemento": menor[1], "id": menor[2],
                "meta": metas[menor[2]], "rotulo": rotulo_caso(metas[menor[2]]),
            }

    df = pd.DataFrame(registros)
    return df, metas, governadores, envelopes, erros


def construir_comparativo(
    df: pd.DataFrame,
    envelopes: dict[tuple[str, int, str], dict[str, Any]],
) -> pd.DataFrame:
    linhas = []
    if df.empty:
        return pd.DataFrame()
    for n in sorted(df["Estrutura"].str.replace("E", "", regex=False).astype(int).unique()):
        for fund in ["F1", "F2"]:
            sub = df[(df["Estrutura"] == f"E{n}") & (df["Fundação"] == fund)]
            if sub.empty:
                continue
            gammas = pd.to_numeric(sub["γz"], errors="coerce")
            betas = pd.to_numeric(sub["β Hogeslag"], errors="coerce")
            lambdas = pd.to_numeric(sub["λmin normalizado(K_E-K_G)"], errors="coerce")
            linha = {
                "Estrutura": f"E{n}",
                "Fundação": fund,
                "Casos resolvidos": int(len(sub)),
                "Casos 2ª ordem instáveis": int((~sub["2ª ordem estável"].astype(bool)).sum()),
                "Casos 2ª ordem quase críticas": int(sub["2ª ordem quase crítica"].astype(bool).sum()),
                "γz máximo finito": float(gammas.max()) if gammas.notna().any() else np.nan,
                "β Hogeslag mínimo": float(betas.min()) if betas.notna().any() else np.nan,
                "Menor autovalor normalizado K_E-K_G": float(lambdas.min()) if lambdas.notna().any() else np.nan,
                "Maior deslocamento relativo (mm)": float(pd.to_numeric(sub["Maior deslocamento relativo 1ª ordem (mm)"]).max()),
                "Maior normal na base (kN)": float(pd.to_numeric(sub["Maior normal na base (kN)"]).max()),
            }
            for ordem in ["1ª ordem", "2ª ordem linearizada"]:
                env = envelopes.get((ordem, n, fund))
                if env:
                    todos_max = [float(np.max(v)) for v in env["max"].values()]
                    todos_min = [float(np.min(v)) for v in env["min"].values()]
                    linha[f"M máximo {ordem} (kN.m)"] = max(todos_max) if todos_max else np.nan
                    linha[f"M mínimo {ordem} (kN.m)"] = min(todos_min) if todos_min else np.nan
                    linha[f"Casos na envoltória {ordem}"] = env["n_casos"]
                else:
                    linha[f"M máximo {ordem} (kN.m)"] = np.nan
                    linha[f"M mínimo {ordem} (kN.m)"] = np.nan
                    linha[f"Casos na envoltória {ordem}"] = 0
            linhas.append(linha)
    return pd.DataFrame(linhas)


def varrer_secao_para_caso(
    meta: dict[str, Any],
    lado_maximo: float,
    incremento: float,
) -> pd.DataFrame:
    """Busca a menor seção no intervalo que mantém K_E-K_G positiva definida.

    É uma triagem de estabilidade global para o cenário fixo, não um dimensionamento:
    não verifica resistência, detalhamento, esbeltez local ou ELU completo.
    """
    lado_inicial = meta["params"].lado_pilar
    if incremento <= 0 or lado_maximo < lado_inicial:
        raise ValueError("Intervalo de seção inválido.")
    valores = list(np.arange(lado_inicial, lado_maximo + incremento * 0.5, incremento))
    if not valores or valores[-1] < lado_maximo - 1e-9:
        valores.append(lado_maximo)
    linhas = []
    encontrou = False
    for lado in valores:
        params_novo = dataclass_replace(meta["params"], lado_pilar=float(lado))
        meta_novo = {**meta, "params": params_novo, "n_pav": params_novo.n_pav}
        caso = executar_meta(meta_novo, calcular_diagramas=False, calcular_modal=False)
        res = caso["resultado"]
        lamb_min = float(res["estabilidade_segunda_ordem"]["lambda_min"])
        quasi = bool(res["estabilidade_segunda_ordem"].get("quase_critica", False))
        estavel = (res["segunda_ordem"] is not None) and (not quasi)
        linhas.append({
            "Lado do pilar (m)": float(lado),
            "Área (m²)": params_novo.area_pilar,
            "Inércia (m⁴)": params_novo.inercia_pilar,
            "Peso próprio (kN/m)": params_novo.peso_proprio_pilar,
            "λmin normalizado(K_E-K_G)": lamb_min,
            "Matriz positiva definida": res["segunda_ordem"] is not None,
            "Quase crítica": quasi,
            "Seção aceita na triagem": estavel,
            "γz": caso["gamma"].get("gamma_z"),
            "β Hogeslag": caso["hogeslag"].get("beta"),
            "Resultado": ("Estável e não quase crítica na triagem" if estavel else
                          ("Quase crítica — não aceita na triagem" if quasi and res["segunda_ordem"] is not None else "Instabilidade indicada")),
        })
        if estavel:
            encontrou = True
            break
    if not encontrou and linhas:
        linhas[-1]["Resultado"] = "Não estabilizou até a seção máxima testada"
    return pd.DataFrame(linhas)


def resumir_envelopes(
    envelopes: dict[tuple[str, int, str], dict[str, Any]],
    metas: dict[str, dict[str, Any]] | None = None,
    df_casos: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Resume extremos por elemento e identifica o cenário e seu estado de 2ª ordem."""
    status_por_id: dict[str, str] = {}
    if df_casos is not None and not df_casos.empty and "ID" in df_casos.columns:
        for _, row in df_casos.iterrows():
            case_id = str(row["ID"])
            if not bool(row.get("2ª ordem estável", False)):
                status_por_id[case_id] = "2ª ordem instável"
            elif bool(row.get("2ª ordem quase crítica", False)):
                cond = row.get("Condicionamento normalizado K_E-K_G", np.nan)
                status_por_id[case_id] = f"quase crítica; cond={float(cond):.2e}"

    def rotulo_com_status(case_id: str) -> str:
        label = rotulo_caso(metas[case_id]) if metas and case_id in metas else case_id
        status = status_por_id.get(case_id)
        return f"{label} [{status}]" if status else label

    linhas = []
    for (ordem, n, fund), env in sorted(envelopes.items()):
        if env["n_casos"] == 0:
            continue
        for (col, trecho), curva_min in env["min"].items():
            curva_max = env["max"][(col, trecho)]
            jmin = int(np.argmin(curva_min))
            jmax = int(np.argmax(curva_max))
            id_min = str(env["id_min"][(col, trecho)][jmin])
            id_max = str(env["id_max"][(col, trecho)][jmax])
            linhas.append({
                "Ordem": ordem,
                "Estrutura": f"E{n}",
                "Fundação": fund,
                "Pilar": col + 1,
                "Trecho": f"{trecho + 1}-{trecho + 2}",
                "Mmin (kN.m)": float(curva_min[jmin]),
                "z de Mmin (m)": float(trecho * H_PAVIMENTO_PADRAO + env["x"][jmin]),
                "ID Mmin": id_min,
                "Caso Mmin": rotulo_com_status(id_min),
                "Mmax (kN.m)": float(curva_max[jmax]),
                "z de Mmax (m)": float(trecho * H_PAVIMENTO_PADRAO + env["x"][jmax]),
                "ID Mmax": id_max,
                "Caso Mmax": rotulo_com_status(id_max),
            })
    return pd.DataFrame(linhas)


def verificar_equilibrio_global(caso: dict[str, Any]) -> dict[str, Any]:
    """Verifica equilíbrio global horizontal/vertical e informa reações por base."""
    p = caso["params"]
    a = caso["acoes"]
    res = caso["resultado"]
    primeira = res["primeira_ordem"]
    reacoes_globais = primeira["reacoes"]

    # Soma de todas as ações horizontais equivalentes, incluindo vento, P*theta
    # nos níveis e as cargas nodais consistentes das imperfeições distribuídas.
    F = res["F"]
    soma_h = float(sum(F[idx_u(nivel)] for nivel in range(p.n_pav + 1)))
    residuo_h = float(reacoes_globais[idx_u(0)] + soma_h)

    soma_normais_base = sum(res["normais"][(c, 0)]["N_base_kN"] for c in range(N_PILARES))
    residuo_v = float(soma_normais_base - a["carga_vertical_total_kN"])

    base_rows = []
    for c in range(N_PILARES):
        item = res.get("elementos_1ordem", {}).get((c, 0))
        if item is None:
            item = recuperar_elementos_revisado(
                p, primeira["d"], res["K_E"], res["K_G"], res["k_local"],
                res["dofs_map"], res["cargas_eq"], res["q_imp"], res["normais"],
                incluir_geometria=False, n_pontos=3,
            )[(c, 0)]
        theta_base = float(primeira["d"][idx_theta(c, 0, p.n_pav)])
        if caso["fundacao"] == "F1":
            m_base_reacao = float(reacoes_globais[idx_theta(c, 0, p.n_pav)])
        else:
            # Momento da mola sobre o nó: ação restauradora oposta à rotação.
            m_base_reacao = -K_ROTACIONAL_PILAR * theta_base
        base_rows.append({
            "Pilar": c + 1,
            "Cortante de base do elemento (kN)": float(item["forcas_extremidade_elasticas"][0]),
            "Momento de base do elemento (kN.m)": float(item["M_base_kNm"]),
            "Normal na base (kN)": float(res["normais"][(c, 0)]["N_base_kN"]),
            "Rotação da base (rad)": theta_base,
            "Momento de reação da fundação (kN.m)": m_base_reacao,
        })
    return {
        "residuo_horizontal_kN": residuo_h,
        "residuo_vertical_kN": residuo_v,
        "reacao_horizontal_base_kN": float(reacoes_globais[idx_u(0)]),
        "soma_acoes_horizontais_kN": soma_h,
        "base_rows": base_rows,
    }


# =============================================================================
# 11. MEMÓRIA DE CÁLCULO EM PDF (REPORTLAB)
# =============================================================================

def gerar_memoria_pdf(payload: dict[str, Any]) -> bytes:
    """Gera memória de cálculo PDF com entradas, hipóteses, casos governantes e resultados."""
    import io
    import os
    from xml.sax.saxutils import escape

    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak,
        Image as RLImage, KeepTogether, LongTable,
    )

    buffer = io.BytesIO()
    page_size = landscape(A4)
    doc = SimpleDocTemplate(
        buffer, pagesize=page_size,
        rightMargin=17 * mm, leftMargin=17 * mm,
        topMargin=18 * mm, bottomMargin=17 * mm,
        title="Memória de cálculo — estabilidade global",
        author="Ferramenta acadêmica de análise estrutural",
    )

    font_regular = "Helvetica"
    font_bold = "Helvetica-Bold"
    paths = [
        ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        ("C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/arialbd.ttf"),
    ]
    for regular_path, bold_path in paths:
        if os.path.exists(regular_path) and os.path.exists(bold_path):
            try:
                pdfmetrics.registerFont(TTFont("MemoriaFont", regular_path))
                pdfmetrics.registerFont(TTFont("MemoriaFont-Bold", bold_path))
                font_regular, font_bold = "MemoriaFont", "MemoriaFont-Bold"
                break
            except Exception:
                pass

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="MemTitle", parent=styles["Title"], fontName=font_bold,
        fontSize=18, leading=22, alignment=TA_CENTER, spaceAfter=8 * mm,
    ))
    styles.add(ParagraphStyle(
        name="MemH1", parent=styles["Heading1"], fontName=font_bold,
        fontSize=13, leading=16, spaceBefore=5 * mm, spaceAfter=2.5 * mm,
    ))
    styles.add(ParagraphStyle(
        name="MemH2", parent=styles["Heading2"], fontName=font_bold,
        fontSize=10.5, leading=13, spaceBefore=3 * mm, spaceAfter=1.5 * mm,
    ))
    styles.add(ParagraphStyle(
        name="MemBody", parent=styles["BodyText"], fontName=font_regular,
        fontSize=8.5, leading=12, spaceAfter=2 * mm,
    ))
    styles.add(ParagraphStyle(
        name="MemSmall", parent=styles["BodyText"], fontName=font_regular,
        fontSize=7, leading=9,
    ))
    styles.add(ParagraphStyle(
        name="MemCell", parent=styles["BodyText"], fontName=font_regular,
        fontSize=6.8, leading=8,
    ))

    story: list[Any] = []

    def p(text: Any, style: str = "MemBody") -> Paragraph:
        return Paragraph(str(text), styles[style])

    def fmt(value: Any, digits: int = 3) -> str:
        try:
            if value is None or pd.isna(value):
                return "—"
            if isinstance(value, (float, np.floating, int, np.integer)):
                if not np.isfinite(float(value)):
                    return "—"
                return f"{float(value):,.{digits}f}".replace(",", "X").replace(".", ",").replace("X", ".")
        except Exception:
            pass
        return escape(str(value))

    def add_table(headers: list[str], rows: list[list[Any]], widths: list[float] | None = None, small: bool = True) -> Table:
        data = [[p(escape(str(h)), "MemCell") for h in headers]]
        for row in rows:
            data.append([p(escape(str(v)), "MemCell") for v in row])
        t = LongTable(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#26374A")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#B7C0CA")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F3F6F8")]),
        ]))
        return t

    def add_figure(fig, width: float = 175 * mm, height: float = 100 * mm) -> None:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        image_buffer = io.BytesIO()
        FigureCanvasAgg(fig).print_png(image_buffer)
        image_buffer.seek(0)
        story.append(RLImage(image_buffer, width=width, height=height))
        story.append(Spacer(1, 2 * mm))
        import matplotlib.pyplot as plt
        plt.close(fig)

    settings = payload["settings"]
    df_comp = payload["comparativo"]
    df_govern = payload["df_governadores"]
    df_env = payload["df_envelopes"]
    df_sections = payload.get("secoes", pd.DataFrame())
    caso = payload["caso_detalhado"]
    params: ParametrosModelo = caso["params"]
    acoes = caso["acoes"]
    res = caso["resultado"]
    gamma = caso["gamma"]
    hog = caso["hogeslag"]
    modal = caso["modal"]

    story.append(Spacer(1, 22 * mm))
    story.append(p("MEMÓRIA DE CÁLCULO", "MemTitle"))
    story.append(p("Estabilidade global de estruturas pré-moldadas", "MemH2"))
    story.append(p("Comparação dos procedimentos de Hogeslag e γz, com análise complementar de segunda ordem geométrica linearizada.", "MemBody"))
    story.append(Spacer(1, 5 * mm))
    story.append(p(f"Data de geração: {datetime.now().strftime('%d/%m/%Y %H:%M')}", "MemBody"))
    story.append(p(
        f"Escopo: {escape(str(settings.get('escopo', 'Análise configurada')))}. "
        f"Cenários resolvidos: {len(payload.get('resumo', pd.DataFrame()))}. "
        f"Cenários com erro: {len(payload.get('erros', []))}. "
        f"Critério de seleção: {escape(payload.get('detail_mode_label', settings.get('criterio_detalhe_label', 'Não informado')))}.",
        "MemBody",
    ))
    story.append(p(
        f"Critério de seleção do caso detalhado: {escape(payload.get('detail_mode_label', settings.get('criterio_detalhe_label', 'Caso representativo')))}. A configuração e o cenário estão identificados nas seções seguintes. Envoltórias regulares excluem casos quase críticos; estes são apresentados separadamente como diagnóstico numérico.",
        "MemSmall",
    ))
    story.append(p("Documento acadêmico gerado automaticamente a partir dos parâmetros e cenários executados no aplicativo. Não constitui memória de dimensionamento normativo nem substitui validação independente.", "MemSmall"))
    story.append(PageBreak())

    story.append(p("1. Objetivo, escopo e modelo", "MemH1"))
    story.append(p(
        "O objetivo é analisar a estabilidade global de pórticos planos de concreto pré-moldado com quatro pilares contínuos e três vãos de 18 m por nível, comparando E1, E2 e E3 e as condições de fundação F1 e F2. As ligações viga-pilar são idealizadas como articuladas; as vigas impõem compatibilidade da translação horizontal dos pilares em cada nível, sem acoplar suas rotações.",
    ))
    story.append(p(
        "Os pilares são representados por elementos de Euler-Bernoulli. A análise de primeira ordem resolve K_E d = F. A análise complementar de segunda ordem geométrica linearizada resolve (K_E - K_G)d = F, com forças normais de primeira ordem mantidas constantes. Esta última não é uma análise materialmente não linear nem uma iteração geométrica incremental completa.",
    ))
    story.append(p("2. Dados de entrada", "MemH1"))
    input_rows = [
        ["Número de pavimentos", str(params.n_pav)],
        ["Número de pilares / vãos", "4 pilares / 3 vãos"],
        ["Comprimento do vão", f"{fmt(params.l_vao)} m"],
        ["Pé-direito", f"{fmt(params.h_pavimento)} m"],
        ["Altura total", f"{fmt(params.altura_total)} m"],
        ["Seção dos pilares", f"{fmt(params.lado_pilar)} × {fmt(params.lado_pilar)} m"],
        ["Área / inércia do pilar", f"{fmt(params.area_pilar, 5)} m² / {fmt(params.inercia_pilar, 6)} m⁴"],
        ["Módulo de elasticidade", f"{fmt(params.ec / 1e6)} GPa"],
        ["Peso específico do concreto", f"{fmt(params.gamma_concreto)} kN/m³"],
        ["Peso próprio linear do pilar", f"{fmt(params.peso_proprio_pilar)} kN/m"],
        ["Carga permanente nas vigas", f"{fmt(params.gk_viga)} kN/m"],
        ["Sobrecarga nas vigas", f"{fmt(params.qk_viga)} kN/m"],
        ["Excentricidade das reações", f"{fmt(params.excentricidade)} m"],
        ["Fator EI nas análises estruturais", fmt(params.fator_ei_1ordem)],
        ["Fator EIeq de Hogeslag", fmt(params.fator_ei_hogeslag)],
        ["Fundação do caso detalhado", caso["fundacao"]],
        ["Desaprumo informado", f"{fmt(settings['theta_mrad'])} mrad"],
        ["Critério/origem do desaprumo", escape(settings.get("theta_fonte", "Não informado"))],
    ]
    story.append(add_table(["Parâmetro", "Valor"], input_rows, [65 * mm, 105 * mm]))

    story.append(p("3. Combinações e ações", "MemH1"))
    comb_rows = []
    for name, c in COMBINACOES.items():
        comb_rows.append([
            name, c["descricao"], fmt(c["gamma_g"], 2), fmt(c["gamma_q"], 2), fmt(c["gamma_w"], 2),
        ])
    story.append(add_table(
        ["Combinação", "Descrição", "γG", "γQ", "γW"], comb_rows,
        [20 * mm, 82 * mm, 20 * mm, 20 * mm, 20 * mm],
    ))
    story.append(p(
        "V1 aplica 80 kN por pavimento. V2 distribui a resultante característica de 80 kN proporcionalmente às cotas dos níveis. Os dois sentidos do vento e, quando informados, os dois sentidos da imperfeição são avaliados separadamente. A sobrecarga pode ser distribuída independentemente nos três vãos de cada pavimento.",
    ))
    story.append(p(
        "Para cada vão carregado, a carga linear de cálculo é qd = γG·gk + γQ·qk (considerando a presença de Q naquele vão). Para viga simplesmente apoiada, a reação característica de cálculo em cada apoio é R = qd·L/2. O momento excêntrico aplicado ao pilar é ±e·R, com sinais opostos nos dois apoios do mesmo vão. O peso próprio dos pilares é wd = γG·γc·A, distribuído ao longo de cada trecho.",
    ))
    if settings.get("incluir_g_favoravel"):
        story.append(p(
            f"Foi incluída uma verificação complementar com γG favorável = {fmt(settings['gamma_g_favoravel'], 2)}, sem Q. Fonte/critério informado: {escape(settings.get('gamma_g_favoravel_fonte', 'Não informado'))}. O coeficiente deve ser confirmado na base normativa adotada; não se presume universal.",
            "MemBody",
        ))

    story.append(p("4. Cenário detalhado e carregamento vertical", "MemH1"))
    story.append(p(f"Caso: {escape(rotulo_caso(payload['main_meta']))}"))
    story.append(p(
        f"Fatores efetivamente usados: γG = {fmt(acoes['gamma_g_utilizado'], 2)}; γQ = {fmt(acoes['gamma_q_utilizado'], 2)}; γW = {fmt(COMBINACOES[caso['combinacao']]['gamma_w'], 2)}. Carga vertical total de cálculo: {fmt(acoes['carga_vertical_total_kN'])} kN.",
    ))
    react_rows = []
    for nivel in range(1, params.n_pav + 1):
        for coluna in range(N_PILARES):
            react_rows.append([
                nivel, f"{nivel * params.h_pavimento:.2f}", coluna + 1,
                fmt(acoes["reacoes"][nivel, coluna]), fmt(acoes["momentos_exc"][nivel, coluna]),
            ])
    story.append(add_table(
        ["Nível", "Cota (m)", "Pilar", "Reação de viga (kN)", "Momento excêntrico (kN.m)"],
        react_rows, [15 * mm, 22 * mm, 15 * mm, 50 * mm, 55 * mm],
    ))
    normal_rows = []
    for (coluna, trecho), ni in res["normais"].items():
        normal_rows.append([
            coluna + 1, f"{trecho + 1}-{trecho + 2}", fmt(ni["N_base_kN"]),
            fmt(ni["N_topo_kN"]), fmt(ni["N_medio_kN"]),
        ])
    story.append(p("Forças normais por trecho", "MemH2"))
    story.append(add_table(
        ["Pilar", "Trecho", "N base (kN)", "N topo (kN)", "N médio (kN)"],
        normal_rows, [25 * mm, 30 * mm, 35 * mm, 35 * mm, 35 * mm],
    ))
    eq = verificar_equilibrio_global(caso)
    story.append(p("Reações na base e equilíbrio global", "MemH2"))
    base_rows = [[
        r["Pilar"], fmt(r["Cortante de base do elemento (kN)"]),
        fmt(r["Momento de base do elemento (kN.m)"]), fmt(r["Normal na base (kN)"]),
        fmt(r["Rotação da base (rad)"], 6), fmt(r["Momento de reação da fundação (kN.m)"]),
    ] for r in eq["base_rows"]]
    story.append(add_table(
        ["Pilar", "V base (kN)", "M base elemento (kN.m)", "N base (kN)", "θ base (rad)", "M reação fundação (kN.m)"],
        base_rows, [18 * mm, 28 * mm, 38 * mm, 28 * mm, 28 * mm, 42 * mm],
    ))
    story.append(p(
        f"Reação horizontal global de base (1ª ordem): {fmt(eq['reacao_horizontal_base_kN'])} kN. "
        f"Resíduo de equilíbrio horizontal: {fmt(eq['residuo_horizontal_kN'], 6)} kN. "
        f"Resíduo de equilíbrio vertical pela soma das normais de base: {fmt(eq['residuo_vertical_kN'], 6)} kN."
    ))

    story.append(p("5. Deslocamentos de primeira ordem e deriva", "MemH1"))
    d1 = res["primeira_ordem"]["d"]
    d2 = res["segunda_ordem"]["d"] if res["segunda_ordem"] is not None else None
    disp_rows = []
    for nivel in range(params.n_pav + 1):
        u1 = d1[idx_u(nivel)]
        dr1 = u1 - d1[idx_u(nivel - 1)] if nivel > 0 else 0.0
        row = [nivel, fmt(nivel * params.h_pavimento), fmt(u1 * 1000, 3), fmt(dr1 * 1000, 3)]
        if d2 is not None:
            u2 = d2[idx_u(nivel)]
            dr2 = u2 - d2[idx_u(nivel - 1)] if nivel > 0 else 0.0
            row.extend([fmt(u2 * 1000, 3), fmt(dr2 * 1000, 3)])
        disp_rows.append(row)
    headers = ["Nível", "Cota (m)", "u1 (mm)", "Δu1 (mm)"]
    if d2 is not None:
        headers += ["u2 (mm)", "Δu2 (mm)"]
    story.append(add_table(headers, disp_rows))

    story.append(p("6. Cálculo de γz", "MemH1"))
    story.append(p("Expressão utilizada: γz = 1 / (1 − ΔMtot,d / M1,tot,d). Os deslocamentos que compõem ΔMtot,d são exclusivamente os de primeira ordem."))
    gamma_rows = [
        ["M1 devido ao vento", f"{fmt(gamma.get('M1_vento_kNm'))} kN.m"],
        ["M1 devido a Pθ das vigas", f"{fmt(gamma.get('M1_imperfeicao_vigas_kNm'))} kN.m"],
        ["M1 devido a Pθ do peso dos pilares", f"{fmt(gamma.get('M1_imperfeicao_pilares_kNm'))} kN.m"],
        ["M1 total (módulo)", f"{fmt(gamma.get('M1_kNm'))} kN.m"],
        ["ΔM das reações das vigas", f"{fmt(gamma.get('DeltaM_vigas_kNm'))} kN.m"],
        ["ΔM do peso próprio dos pilares", f"{fmt(gamma.get('DeltaM_pilares_kNm'))} kN.m"],
        ["ΔM total projetado", f"{fmt(gamma.get('DeltaM_kNm'))} kN.m"],
        ["ΔM/M1", fmt(gamma.get("razao"), 6)],
        ["γz", fmt(gamma.get("gamma_z"), 6)],
        ["Situação", gamma.get("status", "")],
    ]
    story.append(add_table(["Parcela", "Resultado"], gamma_rows, [80 * mm, 90 * mm]))

    story.append(p("7. Procedimento de Hogeslag", "MemH1"))
    story.append(p(
        "São aplicadas as expressões fornecidas no enunciado: Fe = π²(EI)eq/le²; Ff = Kf/Hf; 1/Fref = 1/Fe + 1/Ff; β = Fref/ΣFvd; γH = β/(β−1). Para F1, Ff é tomado como infinito e Fref = Fe.",
    ))
    if "Fe_kN" in hog:
        hog_rows = [
            ["(EI)eq", f"{fmt(hog['EI_eq_kNm2'])} kN.m²"],
            ["le/Htotal adotado", fmt(hog["fator_le"])],
            ["le", f"{fmt(hog['le_m'])} m"],
            ["H_eq (centroide vertical das ações)", f"{fmt(hog['H_eq_centroide_m'])} m"],
            ["Hf adotado", f"{fmt(hog['Hf_m'])} m"],
            ["Fe", f"{fmt(hog['Fe_kN'])} kN"],
            ["Ff", "infinito" if not np.isfinite(hog["Ff_kN"]) else f"{fmt(hog['Ff_kN'])} kN"],
            ["Fref", f"{fmt(hog['Fref_kN'])} kN"],
            ["ΣFvd", f"{fmt(hog['Pvd_total_kN'])} kN"],
            ["β", fmt(hog["beta"])],
            ["γH", fmt(hog.get("gamma_hogeslag"))],
            ["Situação", hog["status"]],
        ]
        story.append(add_table(["Parâmetro", "Resultado"], hog_rows, [80 * mm, 90 * mm]))
    else:
        story.append(p(escape(hog.get("status", "Parâmetros não disponíveis."))))
    story.append(p(
        "Ressalva: le e Hf são equivalências explícitas adotadas no aplicativo. O enunciado alerta que a relação simplificada de fundação foi formulada para carga no topo e que a aplicação a vários níveis requer justificativa/adaptação. Assim, os números de Hogeslag são resultados condicionados a essas hipóteses, não uma validação independente do método.",
    ))

    story.append(p("8. Segunda ordem geométrica linearizada e referência modal", "MemH1"))
    diag = res["estabilidade_segunda_ordem"]
    stable_rows = [
        ["λmin da matriz diagonal-normalizada (adimensional)", fmt(diag["lambda_min"], 8)],
        ["λmax da matriz diagonal-normalizada (adimensional)", fmt(diag["lambda_max"], 8)],
        ["Condicionamento da matriz normalizada", fmt(diag["condicao"], 3)],
        ["Matriz positiva definida", "Sim" if diag["positiva_definida"] else "Não"],
        ["Alerta de quase criticidade", "Sim — cond. > 1e5 (alerta numérico)" if diag.get("quase_critica") else "Não"],
        ["Solução de 2ª ordem", "Disponível" if res["segunda_ordem"] is not None else "Não calculada por perda de positividade"],
        ["Fator crítico modal λcr", fmt(modal.get("lambda_cr"), 6)],
        ["Pcr equivalente modal", f"{fmt(modal.get('Pcr_equiv_kN'))} kN"],
        ["Status modal", modal.get("status", "")],
    ]
    story.append(add_table(["Verificação", "Resultado"], stable_rows, [80 * mm, 90 * mm]))
    story.append(p(
        "Os autovalores acima pertencem à matriz reduzida escalonada por sua diagonal, uma forma adimensional usada para comparar o condicionamento entre graus de liberdade translacionais e rotacionais. O alerta cond. > 1e5 é somente um aviso numérico para resultados muito sensíveis; não é limite normativo nem, isoladamente, prova de instabilidade.",
    ))
    story.append(p(
        "A análise modal por autovalores é uma referência complementar independente: não substitui Hogeslag. A condição positiva definida de K_E-K_G é necessária para a solução linearizada apresentada, mas não equivale à verificação completa de projeto.",
    ))

    story.append(p("9. Momentos fletores e diagramas", "MemH1"))
    elems1 = res.get("elementos_1ordem") or {}
    elems2 = res.get("elementos_2ordem") or {}
    moment_rows = []
    for coluna in range(N_PILARES):
        for trecho in range(params.n_pav):
            item1 = elems1.get((coluna, trecho), {})
            item2 = elems2.get((coluna, trecho), {})
            moment_rows.append([
                coluna + 1, f"{trecho + 1}-{trecho + 2}",
                fmt(item1.get("M_base_kNm")), fmt(item1.get("M_topo_kNm")),
                fmt(item2.get("M_base_kNm")), fmt(item2.get("M_topo_kNm")),
                fmt(item2.get("residuo_momento_topo_relativo"), 5),
            ])
    story.append(add_table(
        ["Pilar", "Trecho", "M base 1ª", "M topo 1ª", "M base 2ª", "M topo 2ª", "Resíduo topo relativo"],
        moment_rows,
        [13 * mm, 18 * mm, 27 * mm, 27 * mm, 27 * mm, 27 * mm, 28 * mm],
    ))
    if elems1:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(N_PILARES, 1, figsize=(8.0, 9.0), sharex=True)
        if N_PILARES == 1:
            axes = [axes]
        for col, ax in enumerate(axes):
            for trecho in range(params.n_pav):
                item = elems1[(col, trecho)]
                z = item["x_m"] + trecho * params.h_pavimento
                ax.plot(item["M_kNm"], z, label=f"Trecho {trecho + 1}-{trecho + 2}")
                if (col, trecho) in elems2:
                    item2 = elems2[(col, trecho)]
                    ax.plot(item2["M_kNm"], z, linestyle="--", label=f"2ª ordem {trecho + 1}-{trecho + 2}")
            ax.axvline(0.0, linewidth=0.7)
            ax.set_ylabel(f"Pilar {col + 1}\nz (m)")
            ax.grid(True, alpha=0.25)
            ax.legend(fontsize=6, loc="best")
        axes[-1].set_xlabel("Momento fletor (kN.m)")
        fig.tight_layout()
        add_figure(fig, width=170 * mm, height=155 * mm)
    story.append(p(
        "Os diagramas de 2ª ordem são recuperados pelo equilíbrio do elemento com a parcela N(x)v'(x). O resíduo entre o momento reconstruído no topo e o momento de extremidade é registrado; resíduos relevantes devem ser investigados antes de usar as curvas como resultados finais.",
    ))

    story.append(p("10. Comparação E1/E2/E3 e F1/F2", "MemH1"))
    if not df_comp.empty:
        cols = [
            "Estrutura", "Fundação", "Casos resolvidos", "Casos 2ª ordem instáveis", "Casos 2ª ordem quase críticas",
            "γz máximo finito", "β Hogeslag mínimo", "Menor autovalor normalizado K_E-K_G",
            "Maior deslocamento relativo (mm)", "Maior normal na base (kN)",
            "M máximo 2ª ordem linearizada (kN.m)", "M mínimo 2ª ordem linearizada (kN.m)",
        ]
        use = df_comp[[c for c in cols if c in df_comp.columns]].copy().fillna("—")
        pretty_rows = []
        for _, rr in use.iterrows():
            pretty = []
            for col in use.columns:
                val = rr[col]
                if isinstance(val, (float, np.floating)):
                    pretty.append(fmt(val, 3))
                else:
                    pretty.append(str(val))
            pretty_rows.append(pretty)
        story.append(add_table(list(use.columns), pretty_rows))
    else:
        story.append(p("Não há dados de comparação disponíveis."))

    story.append(p("11. Casos governantes", "MemH1"))
    if not df_govern.empty:
        rows = []
        for _, r in df_govern.iterrows():
            rows.append([
                r.get("Métrica", ""), r.get("Estrutura", ""), r.get("Fundação", ""),
                fmt(r.get("Valor")), r.get("Caso", ""), r.get("Alerta", ""),
            ])
        story.append(add_table(["Métrica", "E", "Fundação", "Valor", "Caso governante", "Alerta"], rows,
                               [30 * mm, 10 * mm, 18 * mm, 22 * mm, 85 * mm, 50 * mm]))
    else:
        story.append(p("Não foi possível determinar casos governantes."))

    story.append(p("12. Envoltórias de momentos", "MemH1"))
    if not df_env.empty:
        rows = []
        for _, r in df_env.iterrows():
            rows.append([
                r["Ordem"], r["Estrutura"], r["Fundação"], r["Pilar"], r["Trecho"],
                fmt(r["Mmin (kN.m)"]), fmt(r["z de Mmin (m)"]),
                str(r.get("Caso Mmin", ""))[:90],
                fmt(r["Mmax (kN.m)"]), fmt(r["z de Mmax (m)"]),
                str(r.get("Caso Mmax", ""))[:90],
            ])
        story.append(add_table(
            ["Ordem", "E", "F", "Pilar", "Trecho", "Mmin", "z min", "Caso governante Mmin", "Mmax", "z max", "Caso governante Mmax"], rows,
            [23 * mm, 9 * mm, 9 * mm, 12 * mm, 16 * mm, 20 * mm, 16 * mm, 52 * mm, 20 * mm, 16 * mm, 52 * mm],
        ))
    else:
        story.append(p("Não foram geradas envoltórias válidas."))

    env_key = ("2ª ordem linearizada", params.n_pav, caso["fundacao"])
    env_detail = payload["envelopes"].get(env_key)
    if env_detail and env_detail["n_casos"] > 0:
        story.append(p(
            f"A figura seguinte reúne os {env_detail['n_casos']} cenários dessa configuração em que K_E-K_G permaneceu positiva definida. "
            "Esta envoltória contém exclusivamente casos estáveis e não classificados como quase críticos. Casos quase críticos são segregados em tabela/curva diagnóstica e não compõem os valores válidos para comparação de esforços.",
            "MemSmall",
        ))
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(N_PILARES, 1, figsize=(8.0, 9.0), sharex=True)
        if N_PILARES == 1:
            axes = [axes]
        for col, ax in enumerate(axes):
            for trecho in range(params.n_pav):
                key = (col, trecho)
                ax.fill_betweenx(
                    env_detail["x"] + trecho * params.h_pavimento,
                    env_detail["min"][key], env_detail["max"][key], alpha=0.2,
                    label=f"Trecho {trecho + 1}-{trecho + 2}",
                )
                ax.plot(env_detail["min"][key], env_detail["x"] + trecho * params.h_pavimento, linewidth=0.7)
                ax.plot(env_detail["max"][key], env_detail["x"] + trecho * params.h_pavimento, linewidth=0.7)
            ax.axvline(0.0, linewidth=0.7)
            ax.set_ylabel(f"Pilar {col + 1}\nz (m)")
            ax.grid(True, alpha=0.25)
        axes[-1].set_xlabel("Momento fletor (kN.m)")
        fig.suptitle(
            f"Envoltória de 2ª ordem linearizada — E{params.n_pav}/{caso['fundacao']} "
            "(cenários com matriz positiva; inclui casos quase críticos)"
        )
        fig.tight_layout()
        add_figure(fig, width=170 * mm, height=150 * mm)

    # Quase críticos em seção própria: não misturar com envoltórias regulares.
    story.append(p("12A. Diagnóstico separado de casos quase críticos", "MemH1"))
    quasi_rows = []
    for (ordem_q, n_q, f_q), env_q in sorted(payload["envelopes"].items()):
        if ordem_q != "2ª ordem quase crítica" or env_q.get("n_casos", 0) == 0:
            continue
        extremos = []
        for curva in env_q["max"].values():
            if np.isfinite(curva).any(): extremos.append(float(np.nanmax(curva)))
        for curva in env_q["min"].values():
            if np.isfinite(curva).any(): extremos.append(float(np.nanmin(curva)))
        quasi_rows.append([f"E{n_q}", f_q, env_q["n_casos"], fmt(min(extremos) if extremos else np.nan), fmt(max(extremos) if extremos else np.nan)])
    if quasi_rows:
        story.append(p("Os valores abaixo são apenas diagnósticos de sensibilidade numérica. Não devem ser usados como envoltórias regulares nem como esforços de dimensionamento."))
        story.append(add_table(["Estrutura", "Fundação", "Nº casos", "Mínimo diagnóstico (kN.m)", "Máximo diagnóstico (kN.m)"], quasi_rows))
    else:
        story.append(p("Nenhum caso quase crítico foi incluído na varredura."))

    story.append(p("13. Busca preliminar de seção", "MemH1"))
    if not df_sections.empty:
        rows = [[
            r["Estrutura"], r["Fundação"], fmt(r["Lado inicial (m)"]),
            fmt(r["Lado identificado (m)"]), r["Situação"], r["Caso de referência"],
        ] for _, r in df_sections.iterrows()]
        story.append(add_table(
            ["E", "F", "Lado inicial", "Lado identificado", "Situação", "Caso de referência"],
            rows, [12 * mm, 12 * mm, 30 * mm, 35 * mm, 45 * mm, 60 * mm],
        ))
        story.append(p("A busca de seção verifica somente a positividade da matriz global para o cenário informado. Não é dimensionamento completo nem substitui verificações de resistência, esbeltez, detalhamento ou efeitos locais."))
        df_sec_full = payload.get("df_secao_completo", pd.DataFrame())
        if not df_sec_full.empty:
            story.append(p("Resultados completos da varredura de seção", "MemH2"))
            sec_cols = [c for c in ["Estrutura", "Fundação", "Lado do pilar (m)", "λmin normalizado(K_E-K_G)", "Matriz positiva definida", "Quase crítica", "Seção aceita na triagem", "γz", "β Hogeslag", "Resultado"] if c in df_sec_full.columns]
            sec_rows = []
            for _, rr in df_sec_full[sec_cols].iterrows():
                sec_rows.append([fmt(v) if isinstance(v, (float, np.floating)) else str(v) for v in rr.tolist()])
            story.append(add_table(sec_cols, sec_rows))
    else:
        story.append(p("Busca preliminar de seção não solicitada nesta execução."))

    story.append(p("14. Conclusão automática da análise", "MemH1"))
    df_cases = payload.get("resumo", pd.DataFrame())
    if not df_cases.empty:
        n_total = len(df_cases)
        n_inst = int((~df_cases["2ª ordem estável"].astype(bool)).sum())
        n_quasi = int(df_cases["2ª ordem quase crítica"].astype(bool).sum())
        n_regular = n_total - n_inst - n_quasi
        story.append(p(f"Foram processados {n_total} cenários: {n_regular} com matriz de segunda ordem positiva e não quase crítica, {n_quasi} classificados como quase críticos e {n_inst} com perda de positividade/sem solução de segunda ordem estável."))
        if n_inst:
            story.append(p("Conclusão: há configurações para as quais a análise de segunda ordem linearizada não fornece solução estável. Esses casos devem ser tratados como indicação de instabilidade do modelo adotado; não se deve usar deslocamentos ou esforços de segunda ordem desses casos."))
        if n_quasi:
            story.append(p("Conclusão: os casos quase críticos foram segregados das envoltórias regulares. Seus resultados são apresentados apenas como diagnóstico de sensibilidade numérica e exigem conferência independente."))
        if n_regular:
            story.append(p("Os casos regulares podem ser utilizados na comparação acadêmica dentro das hipóteses do modelo, sem dispensar validação independente e verificações normativas."))
        story.append(p(f"Caso detalhado: {escape(rotulo_caso(payload['main_meta']))}. Critério: {escape(payload.get('detail_mode_label', settings.get('criterio_detalhe_label', 'não informado')))}."))
    else:
        story.append(p("Não há resumo de cenários disponível para gerar conclusão automática."))

    story.append(p("15. Hipóteses, limitações e validações pendentes", "MemH1"))
    limits = [
        "γz é apresentado para comparação didática; E1, E2 e E3 têm menos de quatro pavimentos, conforme ressalva do enunciado.",
        "Hogeslag usa le e Hf equivalentes explicitados pelo usuário. A adaptação para cargas em múltiplos níveis e fundação com molas deve ser fundamentada em Hogeslag/El Debs.",
        "A matriz K_E-K_G utiliza forças normais de primeira ordem congeladas; a análise não inclui não linearidade material, grandes rotações, imperfeições locais ou redistribuição não linear de esforços normais.",
        "A flambagem por autovalores é referência independente, não substituição de Hogeslag.",
        "A positividade de K_E-K_G é condição necessária para a solução linearizada, mas não constitui verificação normativa completa.",
        "As envoltórias de momento são amostradas ao longo dos elementos. A resolução espacial da curva deve ser aumentada se for necessário localizar extremos internos com maior precisão.",
        "A verificação complementar de G favorável só é executada quando o usuário informa o coeficiente e sua fonte normativa. A seleção da parcela favorável deve respeitar o efeito estrutural; ações variáveis favoráveis são desconsideradas no cenário complementar.",
        "As envoltórias regulares de 2ª ordem excluem casos classificados como quase críticos pelo limite numérico configurado. Esses casos são apresentados separadamente como diagnóstico e não devem ser usados para dimensionamento.",
        f"Critério de desaprumo informado pelo usuário: {settings.get('theta_fonte', 'Não informado')}. O programa não determina automaticamente o valor normativo de θ.",
        "Antes da entrega, confrontar os resultados com caso de referência independente e revisar a compatibilidade da recuperação dos diagramas, unidades, convenções de sinais e hipóteses de rigidez.",
    ]
    for item in limits:
        story.append(p("• " + escape(item)))

    story.append(Spacer(1, 4 * mm))
    story.append(p(
        "Base do exercício: enunciado 'Estabilidade global de estruturas pré-moldadas — comparação pelos procedimentos de Hogeslag e γz', especialmente as seções 2 a 10. As equivalências e simplificações adicionais estão identificadas neste documento.",
        "MemSmall",
    ))

    def footer(canvas, doc_obj):
        canvas.saveState()
        canvas.setFont(font_regular, 7)
        canvas.setFillColor(colors.HexColor("#5B6570"))
        canvas.drawString(17 * mm, 9 * mm, "Memória de cálculo acadêmica — verificar hipóteses e limitações")
        canvas.drawRightString(page_size[0] - 17 * mm, 9 * mm, f"Página {doc_obj.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return buffer.getvalue()


# Cache de matrizes elásticas: geometria, material e fundação não mudam entre cenários
# de uma mesma configuração. Isso reduz o custo da varredura completa de combinações.
@lru_cache(maxsize=64)
def _montar_matriz_elastica_cache(params: ParametrosModelo, fundacao: str):
    ndof = numero_graus_liberdade(params.n_pav)
    K = np.zeros((ndof, ndof), dtype=float)
    k_local = matriz_rigidez_flexao(params.ei_1ordem, params.h_pavimento)
    dofs_map = {}
    for coluna in range(N_PILARES):
        for trecho in range(params.n_pav):
            dofs = dofs_elemento(coluna, trecho, params.n_pav)
            dofs_map[(coluna, trecho)] = dofs
            K[np.ix_(dofs, dofs)] += k_local
    if fundacao == "F2":
        for coluna in range(N_PILARES):
            dof = idx_theta(coluna, 0, params.n_pav)
            K[dof, dof] += K_ROTACIONAL_PILAR
    elif fundacao != "F1":
        raise ValueError("Fundação deve ser F1 ou F2.")
    return K, k_local, dofs_map

def montar_matriz_elastica(params: ParametrosModelo, fundacao: str):
    return _montar_matriz_elastica_cache(params, fundacao)


# =============================================================================
# 12. NOVA INTERFACE: LOTE COMPLETO, GOVERNANTES, SEÇÕES E MEMÓRIA PDF
# =============================================================================

def reportlab_disponivel() -> bool:
    """Verifica se ReportLab está instalado no mesmo interpretador do Streamlit."""
    try:
        return importlib.util.find_spec("reportlab") is not None
    except (ImportError, ValueError, ModuleNotFoundError):
        return False


def executar_interface() -> None:
    if st is None:
        raise RuntimeError(
            "Streamlit não está instalado. Execute: py -3.13 -m pip install streamlit numpy pandas reportlab matplotlib"
        )
    st.set_page_config(page_title="Estabilidade — Hogeslag, γz e memória", layout="wide")
    st.title("Estabilidade global — estruturas pré-moldadas")
    st.caption("Análise de 1ª ordem, 2ª ordem geométrica linearizada, Hogeslag, γz, envoltórias e memória de cálculo em PDF.")
    st.warning(
        "Ferramenta acadêmica. γz é usado como comparação didática para estruturas com menos de quatro pavimentos. "
        "As equivalências de Hogeslag e a recuperação dos diagramas devem ser justificadas e validadas com referência independente."
    )

    st.sidebar.header("1. Escopo")
    escopo = st.sidebar.radio(
        "Tipo de análise",
        ["Análise completa do exercício", "Análise personalizada"],
        help="A análise completa avalia E1/E2/E3, F1/F2, C1-C4, V1/V2 e todas as distribuições binárias de Q nos vãos e pavimentos.",
    )
    if escopo == "Análise completa do exercício":
        estruturas = [1, 2, 3]
        fundacoes = ["F1", "F2"]
        combinacoes_sel = list(COMBINACOES.keys())
        ventos_sel = ["V1", "V2"]
        varrer_q = True
        st.sidebar.info("Serão avaliadas as seis configurações E1/F1 a E3/F2, as combinações C1-C4, V1/V2 e todas as distribuições parciais de Q.")
    else:
        estruturas = [int(x[1]) for x in st.sidebar.multiselect("Estruturas", ["E1", "E2", "E3"], default=["E1", "E2", "E3"])]
        fundacoes = st.sidebar.multiselect("Fundações", ["F1", "F2"], default=["F1", "F2"])
        combinacoes_sel = st.sidebar.multiselect("Combinações", list(COMBINACOES.keys()), default=["C1", "C2", "C3", "C4"])
        ventos_sel = st.sidebar.multiselect("Distribuições de vento", ["V1", "V2"], default=["V1", "V2"])
        varrer_q = st.sidebar.checkbox("Varredura de todas as distribuições de Q", value=True)
        if not estruturas or not fundacoes or not combinacoes_sel or not ventos_sel:
            st.warning("Selecione pelo menos uma estrutura, fundação, combinação e distribuição de vento.")
            return

    st.sidebar.header("2. Geometria e materiais")
    lado = st.sidebar.number_input("Lado do pilar (m)", min_value=0.20, max_value=1.50, value=0.40, step=0.05)
    ec_gpa = st.sidebar.number_input("Ec (GPa)", min_value=10.0, max_value=60.0, value=30.0, step=1.0)
    gamma_concreto = st.sidebar.number_input("Peso específico do concreto (kN/m³)", min_value=15.0, max_value=30.0, value=25.0, step=0.5)
    gk_viga = st.sidebar.number_input("Gk das vigas (kN/m)", min_value=0.0, max_value=100.0, value=13.0, step=0.5)
    qk_viga = st.sidebar.number_input("Qk das vigas (kN/m)", min_value=0.0, max_value=100.0, value=7.0, step=0.5)
    excentricidade = st.sidebar.number_input("Excentricidade e (m)", min_value=0.0, max_value=2.0, value=0.40, step=0.05)
    fator_ei_1 = st.sidebar.number_input("Fator EI — análise estrutural", min_value=0.05, max_value=1.0, value=1.0, step=0.05)
    fator_ei_hog = st.sidebar.number_input("Fator EIeq — Hogeslag", min_value=0.05, max_value=1.0, value=1.0, step=0.05)

    st.sidebar.header("3. Imperfeições e Hogeslag")
    theta_mrad = st.sidebar.number_input("Desaprumo global θ (mrad)", min_value=0.0, max_value=20.0, value=0.0, step=0.1,
        help="Informe o valor calculado conforme o procedimento normativo adotado. O aplicativo não presume um valor.")
    theta_fonte = st.sidebar.text_input("Origem/critério do desaprumo (obrigatório)", placeholder="Ex.: NBR 6118:2023, item ...; cálculo anexado ...")
    fator_le = st.sidebar.number_input("Relação le/Htotal", min_value=0.50, max_value=4.0, value=2.0, step=0.10)
    hf_manual_ativo = st.sidebar.checkbox("Informar Hf manualmente em F2", value=False)
    hf_manual = st.sidebar.number_input("Hf (m)", min_value=0.10, max_value=100.0, value=6.0, step=0.5) if hf_manual_ativo else None

    st.sidebar.header("4. Verificações complementares")
    incluir_g_fav = st.sidebar.checkbox("Incluir cenário complementar de G favorável", value=False)
    gamma_g_fav = st.sidebar.number_input(
        "Coeficiente γG favorável (confirmar na norma)", min_value=0.1, max_value=1.4, value=1.0, step=0.05,
        disabled=not incluir_g_fav,
        help="O enunciado não fornece esse coeficiente. O valor é editável e deve ser confirmado na edição normativa adotada.",
    )
    gamma_g_fav_fonte = st.sidebar.text_input("Fonte normativa de γG favorável (obrigatória se ativado)", disabled=not incluir_g_fav,
        placeholder="Norma, edição, item/tabela")
    testar_secoes = st.sidebar.checkbox("Executar busca preliminar de seção", value=False)
    lado_max = st.sidebar.number_input("Lado máximo testado (m)", min_value=float(lado), max_value=2.0, value=max(float(lado) + 0.30, float(lado)), step=0.05, disabled=not testar_secoes)
    passo_secao = st.sidebar.number_input("Incremento da seção (m)", min_value=0.01, max_value=0.20, value=0.05, step=0.01, disabled=not testar_secoes)

    mascara_manual = None
    if escopo == "Análise personalizada" and not varrer_q:
        max_n = max(estruturas)
        mascara_manual = np.zeros((max_n, N_VAOS), dtype=bool)
        st.subheader("Distribuição manual da sobrecarga")
        for nivel in range(max_n):
            with st.expander(f"Pavimento {nivel + 1} — cota {(nivel + 1) * H_PAVIMENTO_PADRAO:.0f} m", expanded=(nivel == 0)):
                cols = st.columns(3)
                for vao in range(N_VAOS):
                    mascara_manual[nivel, vao] = cols[vao].checkbox(
                        f"Q no vão {vao + 1}", value=True,
                        key=f"manual_q_{max_n}_{nivel}_{vao}",
                    )

    st.markdown("---")
    executar = st.button("Executar análise", type="primary", use_container_width=True)
    settings = {
        "escopo": escopo,
        "estruturas": estruturas,
        "fundacoes": fundacoes,
        "combinacoes": combinacoes_sel,
        "ventos": ventos_sel,
        "varrer_q": varrer_q,
        "lado": float(lado),
        "ec_gpa": float(ec_gpa),
        "gamma_concreto": float(gamma_concreto),
        "gk_viga": float(gk_viga),
        "qk_viga": float(qk_viga),
        "excentricidade": float(excentricidade),
        "fator_ei_1": float(fator_ei_1),
        "fator_ei_hog": float(fator_ei_hog),
        "theta_mrad": float(theta_mrad),
        "theta": float(theta_mrad) / 1000.0,
        "theta_fonte": theta_fonte.strip(),
        "fator_le": float(fator_le),
        "hf_manual": hf_manual,
        "incluir_g_favoravel": incluir_g_fav,
        "gamma_g_favoravel": float(gamma_g_fav),
        "gamma_g_favoravel_fonte": gamma_g_fav_fonte.strip(),
        "testar_secoes": testar_secoes,
        "lado_max": float(lado_max),
        "passo_secao": float(passo_secao),
        "manual_q_pattern": tuple(bool(v) for v in mascara_manual.ravel()) if mascara_manual is not None else None,
        "criterio_detalhe_label": "Caso representativo (γz mediano; 2ª ordem regular)",
    }

    if executar:
        if theta_mrad <= 0.0:
            st.error("Informe um desaprumo global θ maior que zero, calculado conforme o critério adotado. Para justificar θ = 0, seria necessário documentar formalmente a hipótese; esta versão exige um valor explícito para evitar omissão acidental.")
            return
        if not theta_fonte.strip():
            st.error("Informe a origem normativa ou o critério de cálculo do desaprumo global.")
            return
        if incluir_g_fav and not gamma_g_fav_fonte.strip():
            st.error("Informe a norma, edição e item/tabela que fundamentam γG favorável.")
            return
        params_list = [ParametrosModelo(
            n_pav=n,
            lado_pilar=lado,
            ec=ec_gpa * 1e6,
            gamma_concreto=gamma_concreto,
            gk_viga=gk_viga,
            qk_viga=qk_viga,
            excentricidade=excentricidade,
            fator_ei_1ordem=fator_ei_1,
            fator_ei_hogeslag=fator_ei_hog,
        ) for n in estruturas]
        if varrer_q:
            masks = {n: gerar_mascaras_q(n) for n in estruturas}
        else:
            masks = {n: [mascara_manual[:n, :].copy()] for n in estruturas}

        progresso = st.progress(0.0, text="Montando cenários de cálculo...")
        df, metas, governadores, envelopes, erros = executar_lote_modular(
            parametros=params_list,
            fundacoes=fundacoes,
            combinacoes=combinacoes_sel,
            tipos_vento=ventos_sel,
            mascaras_por_n=masks,
            theta_imp_rad=theta_mrad / 1000.0,
            fator_le_hogeslag=fator_le,
            h_f_manual_m=hf_manual,
            incluir_g_favoravel=incluir_g_fav,
            gamma_g_favoravel=gamma_g_fav,
            callback_progresso=lambda i, total: progresso.progress(i / max(total, 1), text=f"Analisando cenário {i}/{total}"),
        )
        progresso.empty()
        if df.empty:
            st.error("Nenhum cenário foi resolvido. Confira os erros e parâmetros.")
            for erro in erros[:30]:
                st.write(f"- {erro}")
            return

        comparativo = construir_comparativo(df, envelopes)
        df_env = resumir_envelopes(envelopes, metas, df)

        # Caso principal para a memória: maior razão positiva válida; na ausência, maior deriva.
        gov_main = None
        for key, item in governadores.items():
            if key[0] == "razao_delta_m1_max":
                if gov_main is None or item["valor"] > gov_main["valor"]:
                    gov_main = item
        if gov_main is None:
            candidatos = [(k, v) for k, v in governadores.items() if k[0] == "drift_1ordem_max_mm"]
            if candidatos:
                gov_main = max(candidatos, key=lambda kv: kv[1]["valor"])[1]
        if gov_main is None:
            st.error("Não foi possível selecionar um caso detalhado.")
            return
        main_meta = gov_main["meta"]
        caso_detalhado = executar_meta(main_meta, calcular_diagramas=True, calcular_modal=True)

        # Busca de seção para o cenário mais desfavorável de positividade em cada E/F.
        sec_rows = []
        df_secao_completo = pd.DataFrame()
        if testar_secoes:
            for n in estruturas:
                for f in fundacoes:
                    gov = governadores.get(("lambda_min_KE_KG", n, f))
                    if gov is None:
                        continue
                    df_s = varrer_secao_para_caso(gov["meta"], lado_max, passo_secao)
                    if not df_s.empty:
                        primeiro_estavel = df_s[df_s["Seção aceita na triagem"]]
                        sec_ident = float(primeiro_estavel.iloc[0]["Lado do pilar (m)"]) if not primeiro_estavel.empty else np.nan
                        sec_rows.append({
                            "Estrutura": f"E{n}", "Fundação": f,
                            "Lado inicial (m)": lado,
                            "Lado identificado (m)": sec_ident,
                            "Situação": "Encontrada seção com matriz positiva" if np.isfinite(sec_ident) else "Não estabilizou no intervalo",
                            "Caso de referência": gov["rotulo"],
                        })
                        df_s.insert(0, "Fundação", f)
                        df_s.insert(0, "Estrutura", f"E{n}")
                        df_secao_completo = pd.concat([df_secao_completo, df_s], ignore_index=True)
        df_secoes = pd.DataFrame(sec_rows)

        govern_rows = []
        labels = {
            "gamma_z_max": "Maior γz finito",
            "razao_delta_m1_max": "Maior ΔM/M1 válido",
            "drift_1ordem_max_mm": "Maior deslocamento relativo 1ª ordem",
            "N_base_max_kN": "Maior força normal na base",
            "beta_hogeslag_min": "Menor β de Hogeslag",
            "lambda_min_KE_KG": "Menor autovalor normalizado de K_E-K_G",
            "Mmax_1ª ordem": "Maior momento de 1ª ordem",
            "Mmin_1ª ordem": "Menor momento de 1ª ordem",
            "Mmax_2ª ordem linearizada": "Maior momento de 2ª ordem",
            "Mmin_2ª ordem linearizada": "Menor momento de 2ª ordem",
        }
        df_case_lookup = df.set_index("ID", drop=False)
        meta_id_by_label = {rotulo_caso(meta): case_id for case_id, meta in metas.items()}
        for (metric, n, f), item in governadores.items():
            case_id = item.get("id") or meta_id_by_label.get(item.get("rotulo", ""))
            alerta = ""
            if case_id is not None and case_id in df_case_lookup.index:
                linha_case = df_case_lookup.loc[case_id]
                if not bool(linha_case["2ª ordem estável"]):
                    alerta = "2ª ordem instável"
                elif bool(linha_case.get("2ª ordem quase crítica", False)):
                    alerta = f"Quase crítica; cond.={float(linha_case['Condicionamento normalizado K_E-K_G']):.2e}"
            govern_rows.append({
                "Métrica": labels.get(metric, metric),
                "Estrutura": f"E{n}",
                "Fundação": f,
                "Valor": item["valor"],
                "Caso": item["rotulo"],
                "Alerta": alerta,
            })
        df_governadores = pd.DataFrame(govern_rows)

        # Critérios alternativos para escolha do cenário detalhado.
        # Caso representativo = γz mediano entre casos estáveis, não quase críticos e com γz finito.
        selection_candidates: dict[tuple[str, int, str], dict[str, Any]] = {}
        for n in estruturas:
            for f in fundacoes:
                sub = df[(df["Estrutura"] == f"E{n}") & (df["Fundação"] == f)]
                if sub.empty:
                    continue
                regular = sub[(sub["2ª ordem estável"].astype(bool)) & (~sub["2ª ordem quase crítica"].astype(bool))]
                gamma_vals = pd.to_numeric(regular["γz"], errors="coerce")
                valid_gamma = regular[gamma_vals.notna()]
                if not valid_gamma.empty:
                    med = float(pd.to_numeric(valid_gamma["γz"], errors="coerce").median())
                    idx = (pd.to_numeric(valid_gamma["γz"], errors="coerce") - med).abs().idxmin()
                    selection_candidates[("representativo", n, f)] = metas[str(df.loc[idx, "ID"])]
                # Maior gamma_z finito
                gg = pd.to_numeric(sub["γz"], errors="coerce")
                if gg.notna().any():
                    idx = gg.idxmax(); selection_candidates[("maior_gamma", n, f)] = metas[str(df.loc[idx, "ID"])]
                # Mais instável: menor autovalor normalizado, inclusive se negativo.
                ll = pd.to_numeric(sub["λmin normalizado(K_E-K_G)"], errors="coerce")
                if ll.notna().any():
                    idx = ll.idxmin(); selection_candidates[("mais_instavel", n, f)] = metas[str(df.loc[idx, "ID"])]
                # Maior momento válido: extremo de 2ª ordem regular, já sem quase críticos.
                gm = governadores.get(("Mmax_2ª ordem linearizada", n, f))
                gn = governadores.get(("Mmin_2ª ordem linearizada", n, f))
                candidatos_m = [x for x in (gm, gn) if x and x.get("meta")]
                if candidatos_m:
                    escolhido_m = max(candidatos_m, key=lambda x: abs(float(x["valor"])))
                    selection_candidates[("maior_momento", n, f)] = escolhido_m["meta"]

        detalhes_por_config: dict[tuple[int, str], dict[str, Any]] = {}
        for n in estruturas:
            for f in fundacoes:
                escolhido = governadores.get(("razao_delta_m1_max", n, f))
                if escolhido is None:
                    escolhido = governadores.get(("drift_1ordem_max_mm", n, f))
                if escolhido is not None:
                    detalhes_por_config[(n, f)] = escolhido["meta"]

        payload = {
            "settings": settings,
            "detalhes_por_config": detalhes_por_config,
            "selection_candidates": selection_candidates,
            "detail_selection": (main_meta["n_pav"], main_meta["fundacao"]),
            "detail_mode": "representativo",
            "detail_mode_label": "Caso representativo (γz mediano; 2ª ordem regular)",
            "resumo": df,
            "comparativo": comparativo,
            "governadores": governadores,
            "df_governadores": df_governadores,
            "metas": metas,
            "envelopes": envelopes,
            "df_envelopes": df_env,
            "secoes": df_secoes,
            "df_secao_completo": df_secao_completo,
            "erros": erros,
            "main_meta": main_meta,
            "caso_detalhado": caso_detalhado,
        }
        st.session_state["memoria_hogeslag_payload"] = payload
        st.session_state["memoria_hogeslag_settings"] = settings
        st.session_state.pop("memoria_hogeslag_pdf_bytes", None)
        st.session_state.pop("memoria_hogeslag_pdf_key", None)

    payload = st.session_state.get("memoria_hogeslag_payload")
    if payload is None:
        st.info("Configure o escopo e clique em **Executar análise**. A memória de cálculo será exibida após a conclusão da análise.")
        st.subheader("Memória de cálculo em PDF")
        if reportlab_disponivel():
            st.caption("O ReportLab foi encontrado neste ambiente. Execute a análise para habilitar a exportação do memorial.")
        else:
            st.warning("A exportação em PDF ainda não está disponível: o pacote ReportLab não foi encontrado no Python que está executando o Streamlit.")
            st.code("py -3.13 -m pip install reportlab", language="powershell")
            st.caption("Depois da instalação, encerre o Streamlit com Ctrl+C e execute-o novamente.")
        return

    if payload.get("settings") != settings:
        st.warning("Os parâmetros atuais diferem da última execução. Clique em Executar análise para atualizar os resultados e a memória PDF.")
        return

    df = payload["resumo"]
    comparativo = payload["comparativo"]
    df_env = payload["df_envelopes"]
    erros = payload["erros"]
    st.success(f"Análise concluída: {len(df)} cenários resolvidos; {len(erros)} cenários não resolvidos/instáveis.")
    if erros:
        with st.expander(f"Ver {len(erros)} cenários com erro ou instabilidade não resolvida"):
            for erro in erros[:200]:
                st.write(f"- {erro}")

    st.header("1. Comparação entre E1/E2/E3 e F1/F2")
    st.dataframe(comparativo, use_container_width=True, hide_index=True)
    st.download_button(
        "Baixar resumo de todos os cenários (CSV)",
        data=df.to_csv(index=False).encode("utf-8-sig"),
        file_name="resumo_completo_hogeslag_gamma_z.csv",
        mime="text/csv",
    )

    st.header("2. Casos governantes por resultado")
    st.dataframe(payload["df_governadores"], use_container_width=True, hide_index=True)

    opcoes_detalhe = sorted(payload.get("detalhes_por_config", {}).keys())
    if opcoes_detalhe:
        labels_modo = {
            "representativo": "Caso representativo (γz mediano; 2ª ordem regular)",
            "maior_gamma": "Caso de maior γz finito",
            "mais_instavel": "Caso mais instável (menor autovalor de K_E−K_G)",
            "maior_momento": "Caso de maior |momento| válido de 2ª ordem",
        }
        modo_atual = st.selectbox("Critério de seleção do caso detalhado", list(labels_modo),
            format_func=lambda x: labels_modo[x], key="modo_detalhe_hog")
        candidatos_modo = {k[1:]: v for k, v in payload.get("selection_candidates", {}).items() if k[0] == modo_atual}
        opcoes_modo = sorted(candidatos_modo)
        if not opcoes_modo:
            st.warning("Não há caso disponível para esse critério. Escolha outro critério de seleção.")
        else:
            selecao_atual = payload.get("detail_selection", opcoes_modo[0])
            if selecao_atual not in opcoes_modo: selecao_atual = opcoes_modo[0]
            escolha_detalhe = st.selectbox("Estrutura/fundação para detalhamento", options=opcoes_modo,
                index=opcoes_modo.index(selecao_atual), format_func=lambda x: f"E{x[0]} — {x[1]}")
            meta_detalhe = candidatos_modo[escolha_detalhe]
            meta_atual_id = rotulo_caso(payload["main_meta"])
            if escolha_detalhe != payload.get("detail_selection") or modo_atual != payload.get("detail_mode") or rotulo_caso(meta_detalhe) != meta_atual_id:
                with st.spinner("Calculando o detalhamento do cenário selecionado..."):
                    payload["caso_detalhado"] = executar_meta(meta_detalhe, calcular_diagramas=True, calcular_modal=True)
                payload["main_meta"] = meta_detalhe
                payload["detail_selection"] = escolha_detalhe
                payload["detail_mode"] = modo_atual
                payload["detail_mode_label"] = labels_modo[modo_atual]
                st.session_state["memoria_hogeslag_payload"] = payload
                st.session_state.pop("memoria_hogeslag_pdf_bytes", None)
                st.session_state.pop("memoria_hogeslag_pdf_key", None)

    st.header("3. Memória de cálculo — caso selecionado")
    caso = payload["caso_detalhado"]
    g = caso["gamma"]
    h = caso["hogeslag"]
    m = caso["modal"]
    st.caption(rotulo_caso(payload["main_meta"]))
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("γz", f"{g['gamma_z']:.4f}" if g.get("gamma_z") is not None else "Não aplicável")
    k2.metric("β Hogeslag", fmt_ui(h.get("beta"), 4))
    k3.metric("λcr modal", fmt_ui(m.get("lambda_cr"), 4))
    k4.metric("2ª ordem", "Estável" if caso["resultado"]["segunda_ordem"] is not None else "Não estável")
    st.write(g["status"])
    diag_case = caso["resultado"]["estabilidade_segunda_ordem"]
    if diag_case.get("quase_critica"):
        st.warning(
            "A matriz K_E-K_G permanece positiva definida, mas está numericamente quase crítica "
            f"(condicionamento diagonal-normalizado = {diag_case['condicao']:.3e}). "
            "Os deslocamentos e momentos de 2ª ordem são muito sensíveis e devem ser interpretados com cautela."
        )
    st.write(h.get("status", "Hogeslag não disponível."))
    if "Fe_kN" in h:
        st.write(
            f"Fe = {h['Fe_kN']:.2f} kN; Ff = {fmt_ui(h['Ff_kN'], 2)} kN; "
            f"Fref = {h['Fref_kN']:.2f} kN; β = {h['beta']:.4f}; "
            f"γH = {fmt_ui(h.get('gamma_hogeslag'), 4)}."
        )

    # Displacement table
    p = caso["params"]
    res = caso["resultado"]
    d1 = res["primeira_ordem"]["d"]
    d2 = res["segunda_ordem"]["d"] if res["segunda_ordem"] is not None else None
    linhas_d = []
    for nivel in range(p.n_pav + 1):
        row = {
            "Nível": nivel,
            "Cota (m)": nivel * p.h_pavimento,
            "u 1ª ordem (mm)": d1[idx_u(nivel)] * 1000.0,
            "Δu 1ª ordem (mm)": ((d1[idx_u(nivel)] - d1[idx_u(nivel - 1)]) * 1000.0) if nivel > 0 else 0.0,
        }
        if d2 is not None:
            row["u 2ª ordem (mm)"] = d2[idx_u(nivel)] * 1000.0
            row["Δu 2ª ordem (mm)"] = ((d2[idx_u(nivel)] - d2[idx_u(nivel - 1)]) * 1000.0) if nivel > 0 else 0.0
        linhas_d.append(row)
    st.subheader("Deslocamentos absolutos e relativos")
    st.dataframe(pd.DataFrame(linhas_d), use_container_width=True, hide_index=True)

    st.subheader("Ações verticais, momentos excêntricos e forças normais")
    reacoes_rows = []
    for nivel in range(1, p.n_pav + 1):
        for c in range(N_PILARES):
            reacoes_rows.append({
                "Nível": nivel, "Pilar": c + 1,
                "R vigas (kN)": caso["acoes"]["reacoes"][nivel, c],
                "M excêntrico (kN.m)": caso["acoes"]["momentos_exc"][nivel, c],
            })
    st.dataframe(pd.DataFrame(reacoes_rows), use_container_width=True, hide_index=True)
    st.dataframe(pd.DataFrame([
        {"Pilar": c + 1, "Trecho": f"{t + 1}-{t + 2}", **v}
        for (c, t), v in res["normais"].items()
    ]), use_container_width=True, hide_index=True)
    eq_check = verificar_equilibrio_global(caso)
    st.subheader("Reações de apoio e equilíbrio")
    st.dataframe(pd.DataFrame(eq_check["base_rows"]), use_container_width=True, hide_index=True)
    st.write(
        f"Reação horizontal global na base: {eq_check['reacao_horizontal_base_kN']:.4f} kN | "
        f"Resíduo horizontal: {eq_check['residuo_horizontal_kN']:.3e} kN | "
        f"Resíduo vertical: {eq_check['residuo_vertical_kN']:.3e} kN."
    )

    st.subheader("Diagramas de momentos do caso principal")
    elems = res["elementos_1ordem"]
    st.caption("A curva de segunda ordem é exibida somente quando K_E-K_G permanece positiva definida.")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(N_PILARES, 1, figsize=(8, 9), sharex=True)
    if N_PILARES == 1:
        axes = [axes]
    for c, ax in enumerate(axes):
        for t in range(p.n_pav):
            item = elems[(c, t)]
            z = item["x_m"] + t * p.h_pavimento
            ax.plot(item["M_kNm"], z, label=f"1ª ordem — trecho {t + 1}-{t + 2}")
            if res.get("elementos_2ordem") is not None:
                item2 = res["elementos_2ordem"][(c, t)]
                ax.plot(item2["M_kNm"], z, linestyle="--", label=f"2ª ordem — trecho {t + 1}-{t + 2}")
                if item2["residuo_momento_topo_relativo"] > 0.01:
                    st.warning(
                        f"Pilar {c + 1}, trecho {t + 1}-{t + 2}: resíduo de recuperação do momento no topo "
                        f"de {item2['residuo_momento_topo_relativo'] * 100:.2f}%. Verificar a discretização/modelo."
                    )
        ax.axvline(0, linewidth=0.7)
        ax.grid(True, alpha=0.25)
        ax.set_ylabel(f"Pilar {c + 1}\nz (m)")
        ax.legend(fontsize=7)
    axes[-1].set_xlabel("Momento (kN.m)")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.header("4. Envoltórias dos cenários analisados")
    st.info("As envoltórias regulares de 2ª ordem excluem casos quase críticos. Estes são apresentados separadamente como diagnóstico numérico e não devem ser usados para dimensionamento.")
    if payload["envelopes"]:
        opcoes_env = sorted(payload["envelopes"].keys(), key=lambda x: (x[1], x[2], x[0]))
        labels_env = [f"{o} — E{n}/{f}" for o, n, f in opcoes_env]
        escolha_env = st.selectbox("Configuração da envoltória", range(len(opcoes_env)), format_func=lambda i: labels_env[i])
        ordem, n_env, f_env = opcoes_env[escolha_env]
        env = payload["envelopes"][(ordem, n_env, f_env)]
        fig2, axes2 = plt.subplots(N_PILARES, 1, figsize=(8, 9), sharex=True)
        if N_PILARES == 1:
            axes2 = [axes2]
        for c, ax in enumerate(axes2):
            for t in range(n_env):
                key = (c, t)
                ax.fill_betweenx(env["x"] + t * H_PAVIMENTO_PADRAO, env["min"][key], env["max"][key], alpha=0.2)
                ax.plot(env["min"][key], env["x"] + t * H_PAVIMENTO_PADRAO, linewidth=0.8)
                ax.plot(env["max"][key], env["x"] + t * H_PAVIMENTO_PADRAO, linewidth=0.8)
            ax.axvline(0, linewidth=0.7)
            ax.grid(True, alpha=0.25)
            ax.set_ylabel(f"Pilar {c + 1}\nz (m)")
        axes2[-1].set_xlabel("Momento fletor (kN.m)")
        fig2.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)
        if ordem == "2ª ordem linearizada":
            st.caption(
                f"Envoltória formada por {env['n_casos']} cenários com K_E-K_G positiva definida. "
                "Casos quase críticos também entram na envoltória e aparecem sinalizados nas tabelas; confira-os com cautela."
            )
        st.dataframe(df_env[(df_env["Ordem"] == ordem) & (df_env["Estrutura"] == f"E{n_env}") & (df_env["Fundação"] == f_env)], use_container_width=True, hide_index=True)
        quasi_df = df_env[(df_env["Ordem"] == "2ª ordem quase crítica") & (df_env["Estrutura"] == f"E{n_env}") & (df_env["Fundação"] == f_env)]
        if not quasi_df.empty:
            with st.expander("Ver diagnóstico separado dos casos quase críticos"):
                st.dataframe(quasi_df, use_container_width=True, hide_index=True)
        st.download_button(
            "Baixar tabela completa das envoltórias (CSV)",
            data=df_env.to_csv(index=False).encode("utf-8-sig"),
            file_name="envoltorias_completas_momentos.csv",
            mime="text/csv",
        )
    else:
        st.warning("Não foram geradas envoltórias válidas.")

    if not payload["secoes"].empty:
        st.header("5. Busca preliminar de alteração da seção")
        st.dataframe(payload["secoes"], use_container_width=True, hide_index=True)
        st.caption("A busca verifica positividade de K_E-K_G para o cenário crítico de cada E/F. Não é dimensionamento completo.")
        if not payload["df_secao_completo"].empty:
            st.dataframe(payload["df_secao_completo"], use_container_width=True, hide_index=True)

    st.header("6. Exportação")
    if not reportlab_disponivel():
        st.error("Não foi possível habilitar a memória em PDF porque ReportLab não está instalado no interpretador que executa o Streamlit.")
        st.markdown("Instale a dependência no mesmo Python usado para iniciar o aplicativo:")
        st.code("py -3.13 -m pip install reportlab", language="powershell")
        st.caption("Após instalar, interrompa o servidor Streamlit (Ctrl+C) e execute novamente o comando de inicialização.")
    else:
        pdf_key = (repr(payload.get("settings", {})), repr(payload.get("detail_selection")))
        if (st.session_state.get("memoria_hogeslag_pdf_key") != pdf_key
                or st.session_state.get("memoria_hogeslag_pdf_bytes") is None):
            try:
                with st.spinner("Gerando a memória de cálculo em PDF..."):
                    pdf_bytes = gerar_memoria_pdf(payload)
                st.session_state["memoria_hogeslag_pdf_bytes"] = pdf_bytes
                st.session_state["memoria_hogeslag_pdf_key"] = pdf_key
            except ModuleNotFoundError as exc:
                st.error(f"Dependência ausente para gerar o PDF: {exc}")
                st.code("py -3.13 -m pip install reportlab", language="powershell")
            except Exception as exc:
                st.error(f"Falha ao montar a memória de cálculo: {exc}")
                st.exception(exc)
        pdf_bytes = st.session_state.get("memoria_hogeslag_pdf_bytes")
        if pdf_bytes:
            st.success(f"Memória de cálculo pronta: {len(pdf_bytes):,} bytes.".replace(",", "."))
            st.download_button(
                "Baixar memória de cálculo em PDF",
                data=pdf_bytes,
                file_name="memoria_calculo_estabilidade_hogeslag_gamma_z.pdf",
                mime="application/pdf",
                type="primary",
                use_container_width=True,
            )
    st.download_button(
        "Baixar tabela de casos governantes (CSV)",
        data=payload["df_governadores"].to_csv(index=False).encode("utf-8-sig"),
        file_name="casos_governantes_estabilidade.csv",
        mime="text/csv",
    )

    st.header("7. Critérios e limitações para o relatório")
    st.markdown(
        """
        - **γz:** calculado somente com deslocamentos de primeira ordem; comparar didaticamente, pois as estruturas têm menos de quatro pavimentos.
        - **Hogeslag:** expressões do enunciado, mas `le` e `Hf` são equivalências que precisam de fundamentação bibliográfica para vários níveis.
        - **Segunda ordem:** análise geométrica linearizada com `K_E-K_G` e forças normais congeladas; não é análise não linear completa. A matriz é escalonada diagonalmente para avaliar a inércia; condicionamento normalizado acima de 1e5 gera alerta numérico, não critério normativo.
        - **Envoltórias:** consideram os cenários executados e identificam a combinação e a distribuição de cargas responsáveis pelos extremos amostrados.
        - **Seção:** a busca identifica uma seção que mantém a matriz global positiva no cenário verificado; não verifica resistência nem detalhamento.
        - **Ações permanentes favoráveis:** só são avaliadas no cenário complementar se o usuário informar o fator apropriado da norma adotada.
        """
    )


def fmt_ui(value: Any, digits: int = 3) -> str:
    try:
        if value is None or not np.isfinite(float(value)):
            return "Não definido"
        return f"{float(value):.{digits}f}"
    except Exception:
        return "Não definido"


if __name__ == "__main__":
    executar_interface()
