import streamlit as st
import numpy as np

# ==========================================
# CONFIGURAÇÃO DA INTERFACE (STREAMLIT)
# ==========================================
st.set_page_config(page_title="Análise de Estabilidade - Pré-Moldados", layout="wide")
st.title("Análise de Estabilidade Global de Estruturas Pré-Moldadas")
st.markdown("Comparativo de Pórticos Planos (Método $\gamma_z$) - Vento V2 e Combinação C2")

# --- BARRA LATERAL (INPUTS) ---
st.sidebar.header("Configurações do Modelo")
estrutura = st.sidebar.selectbox("Estrutura:", ["E1 (1 Pavimento)", "E2 (2 Pavimentos)", "E3 (3 Pavimentos)"])
fundacao = st.sidebar.radio("Condição de Fundação:", ["F1 (Engaste Perfeito)", "F2 (Mola Rotacional)"])

num_pav = int(estrutura[1]) # Extrai o número de pavimentos (1, 2 ou 3)

st.sidebar.subheader("Aplicação da Sobrecarga ($Q_k$)")
st.sidebar.caption("Ative para aplicar a sobrecarga no respectivo nível:")
presenca_qk = []
for i in range(num_pav):
    presenca_qk.append(st.sidebar.checkbox(f"Carregar Nível {i+1} (Cota {(i+1)*6}m)", value=True))

# ==========================================
# NÚCLEO DE CÁLCULO ESTRUTURAL
# ==========================================
# 1. Parâmetros Geométricos e Físicos
E_c = 30e6          # Módulo de elasticidade (kN/m2)
I_pilar = (0.40**4) / 12 
I_eq = 4 * I_pilar  # 4 pilares em paralelo
H = 6.0             # Pé-direito (m)
EI = (E_c * I_eq) / 3 # Rigidez secante reduzida para estabilidade global

# 2. Esforços Verticais (Gk, Qk, Peso Próprio)
R_gk_d = 1.4 * (13 * 18 / 2)  # Reação permanente (kN)
R_qk_d = 1.4 * (7 * 18 / 2)   # Reação de sobrecarga (kN)
P_pilar_d = 1.4 * 24          # Peso próprio do trecho (kN)
e_exc = 0.40                  # Excentricidade (m)

# 3. Forças Horizontais (Vento V2) - Valores característicos por nível
vento_v2_dict = {
    1: [80.0],
    2: [26.67, 53.33],
    3: [13.33, 26.67, 40.00]
}
W_d = 0.84 * np.array(vento_v2_dict[num_pav]) # Ponderação Combinação C2

# 4. Cálculo das Cargas Verticais por Pavimento (F_vd_i)
F_vd_pavimentos = np.zeros(num_pav)
momentos_ext = np.zeros(num_pav)
momentos_int = np.zeros(num_pav)

for i in range(num_pav):
    tem_q = presenca_qk[i]
    # Reações nos pilares isolados
    N_ext = P_pilar_d + R_gk_d + (R_qk_d if tem_q else 0)
    N_int = P_pilar_d + 2*R_gk_d + (2*R_qk_d if tem_q else 0)
    F_vd_pavimentos[i] = 2 * N_ext + 2 * N_int # Carga total no andar
    
    # Momentos excêntricos
    momentos_ext[i] = (R_gk_d + (R_qk_d if tem_q else 0)) * e_exc
    momentos_int[i] = (R_qk_d * e_exc) if tem_q else 0.0

# 5. Montagem da Matriz de Rigidez Global (Pórtico Equivalente)
n_nos = num_pav + 1
n_gl = n_nos * 2 # 2 graus de liberdade por nó (Translação X, Rotação Z)
K_global = np.zeros((n_gl, n_gl))

def rigidez_flexao(EI, L):
    return (EI / L**3) * np.array([
        [12,   6*L,     -12,  6*L],
        [6*L,  4*L**2,  -6*L, 2*L**2],
        [-12, -6*L,      12, -6*L],
        [6*L,  2*L**2,  -6*L, 4*L**2]
    ])

# Adicionando a rigidez de cada tramo de pilar
for i in range(num_pav):
    idx = i * 2
    K_global[idx:idx+4, idx:idx+4] += rigidez_flexao(EI, H)

# 6. Condições de Contorno e Solução do Sistema
if "F2" in fundacao:
    K_global[1, 1] += 300000 # Mola de 75.000 kN.m/rad * 4 pilares
    gl_livres = list(range(1, n_gl)) # Apenas translação na base impedida
else:
    gl_livres = list(range(2, n_gl)) # Translação e rotação na base impedidos

# Vetor de Forças
F_global = np.zeros(len(gl_livres))
# Mapeando os esforços de vento nos graus de liberdade de translação adequados
for i in range(num_pav):
    if "F2" in fundacao:
        gl_translacao = (i + 1) * 2 - 1 
    else:
        gl_translacao = (i + 1) * 2 - 2
    F_global[gl_translacao] = W_d[i]

# Inversão e cálculo dos deslocamentos
K_livre = K_global[np.ix_(gl_livres, gl_livres)]
deslocamentos_livres = np.linalg.solve(K_livre, F_global)

# Recuperando os deslocamentos de translação por pavimento
desloc_translacao = np.zeros(num_pav)
for i in range(num_pav):
    if "F2" in fundacao:
        desloc_translacao[i] = deslocamentos_livres[(i + 1) * 2 - 1]
    else:
        desloc_translacao[i] = deslocamentos_livres[(i + 1) * 2 - 2]

# 7. Cálculo do Gama-z
alturas = np.array([(i+1)*H for i in range(num_pav)])
M1_tot_d = np.sum(W_d * alturas)
Delta_M_tot_d = np.sum(F_vd_pavimentos * desloc_translacao)
Gama_z = 1 / (1 - (Delta_M_tot_d / M1_tot_d))

# ==========================================
# EXIBIÇÃO DE RESULTADOS NA INTERFACE
# ==========================================
col1, col2 = st.columns(2)

with col1:
    st.subheader("Parâmetros de Estabilidade")
    st.metric(label="Momento Tombamento (M1,tot,d)", value=f"{M1_tot_d:.2f} kNm")
    st.metric(label="Momento de 2ª Ordem (ΔM,tot,d)", value=f"{Delta_M_tot_d:.2f} kNm")
    st.metric(label="Coeficiente Gama-z", value=f"{Gama_z:.3f}", delta="Crítico (>1.1)" if Gama_z > 1.1 else "OK", delta_color="inverse")

with col2:
    st.subheader("Deslocamentos e Esforços")
    for i in range(num_pav):
        st.write(f"**Nível {i+1} (Cota {alturas[i]}m):**")
        st.write(f"- Deslocamento Lateral: `{desloc_translacao[i]*1000:.2f} mm`")
        st.write(f"- Momentos Excêntricos: Ext = `{momentos_ext[i]:.2f} kNm` | Int = `{momentos_int[i]:.2f} kNm`")
        st.write(f"- Força Normal Acumulada ($\Sigma F_{{vd}}$ do andar): `{F_vd_pavimentos[i]:.2f} kN`")
        st.divider()

# 7. Cálculo do Gama-z
alturas = np.array([(i+1)*H for i in range(num_pav)])
M1_tot_d = np.sum(W_d * alturas)
Delta_M_tot_d = np.sum(F_vd_pavimentos * desloc_translacao)
Gama_z = 1 / (1 - (Delta_M_tot_d / M1_tot_d))

# ==========================================
# 8. MÉTODO DE HOGESLAG (Adaptado para múltiplos níveis)
# ==========================================
# Carga vertical total do pórtico
F_vd_total = np.sum(F_vd_pavimentos)

# Altura Equivalente (Heq) - Média ponderada das cargas pelas cotas
H_eq = np.sum(F_vd_pavimentos * alturas) / F_vd_total

# Comprimento de Flambagem (le). Para balanço puro (F1): le = 2 * Heq.
# (A influência da mola em F2 entrará no termo F_f)
l_e = 2 * H_eq

# Força Crítica de Euler (Fe)
F_e = (np.pi**2 * EI) / (l_e**2)

# Fator de Fundação (Ff)
if "F2" in fundacao:
    K_f_total = 300000 # Rigidez rotacional total dos 4 pilares
    F_f = K_f_total / H_eq
else:
    F_f = float('inf') # Engaste perfeito

# Força de Referência (F_ref)
if F_f == float('inf'):
    F_ref = F_e
else:
    F_ref = 1 / ((1 / F_e) + (1 / F_f))

# Parâmetros beta e gama
beta = F_ref / F_vd_total
gama_hogeslag = beta / (beta - 1) if beta > 1 else float('inf')

# ==========================================
# EXIBIÇÃO DE RESULTADOS NA INTERFACE
# ==========================================
st.markdown("---")
st.header("Análise Comparativa de Estabilidade")

# Painel de Indicadores Principais
col1, col2, col3 = st.columns(3)
col1.metric(label="Coeficiente Gama-z ($\gamma_z$)", value=f"{Gama_z:.3f}")
col2.metric(label="Fator Hogeslag ($\gamma$)", value=f"{gama_hogeslag:.3f}")
col3.metric(label="Carga Vertical Total ($\Sigma F_{vd}$)", value=f"{F_vd_total:.2f} kN")

st.markdown("---")

col_esq, col_dir = st.columns(2)

with col_esq:
    st.subheader("Parâmetros do Método de Hogeslag")
    st.write(f"- Altura Equivalente Ponderada ($H_{{eq}}$): `{H_eq:.2f} m`")
    st.write(f"- Comprimento de Flambagem ($l_e$): `{l_e:.2f} m`")
    st.write(f"- Carga Crítica de Euler ($F_e$): `{F_e:.2f} kN`")
    if "F2" in fundacao:
        st.write(f"- Fator de Fundação ($F_f$): `{F_f:.2f} kN`")
    st.write(f"- Força de Referência ($F_{{ref}}$): `{F_ref:.2f} kN`")
    st.write(f"- Fator de Segurança ($\beta$): `{beta:.2f}`")

with col_dir:
    st.subheader("Deslocamentos e Momentos de 1ª Ordem")
    for i in range(num_pav):
        st.write(f"**Nível {i+1} (Cota {alturas[i]}m):**")
        st.write(f"- Deslocamento Lateral: `{desloc_translacao[i]*1000:.2f} mm`")
        st.write(f"- Momentos Excêntricos: Ext = `{momentos_ext[i]:.2f} kNm` | Int = `{momentos_int[i]:.2f} kNm`")
        
    st.info(f"O Momento de Tombamento Global foi amplificado em aproximadamente **{(max(Gama_z, gama_hogeslag) - 1)*100:.1f}%** devido aos efeitos de 2ª ordem.")
    # ==========================================
# 9. DIAGNÓSTICO COMPARATIVO E DISCUSSÃO
# ==========================================
st.markdown("---")
st.header("Diagnóstico Comparativo: Hogeslag vs Gama-z")

# Cálculo da diferença relativa entre os métodos
if beta <= 1:
    st.error("⚠️ **INSTABILIDADE DETECTADA:** A carga vertical superou a carga crítica de flambagem do pórtico (beta <= 1). A estrutura é instável e os métodos simplificados não são aplicáveis. É necessário aumentar a seção transversal dos pilares (superior a 40x40 cm).")
else:
    diferenca_percentual = abs(Gama_z - gama_hogeslag) / min(Gama_z, gama_hogeslag) * 100
    
    st.write(f"Nesta simulação para a **{estrutura}** com **{fundacao}**, o método do $\gamma_z$ resultou em `{Gama_z:.3f}` e o método de Hogeslag em `{gama_hogeslag:.3f}` (Diferença de **{diferenca_percentual:.1f}%**).")
    
    with st.expander("Ver Interpretação Analítica e Limitações Normativas", expanded=True):
        st.markdown(f"""
        **1. Impacto da Altura (Transição entre E1, E2 e E3):**
        À medida que o pórtico cresce, a força vertical total ($\Sigma F_{{vd}}$) acumulada na base aumenta substancialmente, enquanto a esbeltez global reduz a rigidez do sistema. Isso afasta o pórtico da sua condição de estabilidade ideal, inflando ambos os coeficientes.
        
        **2. Impacto da Fundação (Engaste F1 vs. Mola F2):**
        A introdução da mola rotacional (F2) introduz um fator de relaxamento severo. 
        * No **Método de Hogeslag**, isso insere o termo restritivo $F_f$, reduzindo a força de referência $F_{{ref}}$ da estrutura e disparando o multiplicador $\gamma$. 
        * No **Gama-z**, a rotação da base adiciona uma parcela de corpo rígido ao deslocamento lateral, elevando os momentos de 2ª ordem ($\Delta M_{{tot,d}}$).
        
        **3. Divergência Teórica e Limites do $\gamma_z$:**
        O $\gamma_z$ foi originalmente calibrado pela NBR 6118 para edifícios de múltiplos pavimentos (geralmente $\ge 4$), assumindo um comportamento global contínuo. Ao aplicá-lo em estruturas muito baixas (como a E1 de apenas 1 pavimento), estamos extrapolando o seu domínio normativo ótimo. Nestes casos, o **Método de Hogeslag** costuma ser mais rigoroso e representativo, pois a sua formulação captura a estabilidade através da energia de deformação e da carga crítica de Euler pura, sendo extremamente sensível às condições de contorno da base.
        """)