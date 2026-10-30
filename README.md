PROGRAMA DE ESTABILIDADE GLOBAL — VERSÃO 1.3

ARQUIVO PRINCIPAL
  tarefa_5_hogeslag_1_3_corrigido.py

INSTALAÇÃO (Python 3.13 no Windows / PowerShell)
  py -3.13 -m pip install -r requirements_hogeslag_1_3.txt

EXECUÇÃO
  py -3.13 -m streamlit run tarefa_5_hogeslag_1_3_corrigido.py

ALTERAÇÕES PRINCIPAIS
1. Envoltórias regulares de 2ª ordem excluem cenários classificados como quase críticos.
   Os casos quase críticos são armazenados e exibidos em grupo separado, como diagnóstico.
2. Seleção do caso detalhado por: caso representativo (mediana de gamma_z entre casos
   regulares), maior gamma_z finito, caso mais instável (menor autovalor normalizado) ou
   maior módulo de momento válido de 2ª ordem.
3. A interface exige desaprumo global maior que zero e a indicação da origem/critério adotado.
   O programa não escolhe automaticamente o valor normativo.
4. A verificação complementar de G favorável exige coeficiente e fonte normativa informados
   pelo usuário; a ação variável Q é retirada dessa verificação complementar.
5. Busca progressiva de seção: registra cada seção testada e só aceita como resultado da
   triagem uma seção com matriz positiva definida e não classificada como quase crítica.
6. A memória PDF inclui conclusão automática com contagem de cenários regulares, quase críticos
   e instáveis, critério de seleção do caso detalhado e hipóteses/limitações.

OBSERVAÇÕES TÉCNICAS IMPORTANTES
- O limite numérico de quase criticidade é um critério computacional, não um limite normativo.
- A análise K_E-K_G é segunda ordem geométrica linearizada com forças normais de 1ª ordem
  mantidas constantes; não é análise não linear incremental completa.
- A triagem de seção não é dimensionamento. Não verifica resistência, armaduras, esbeltez
  local, efeitos locais de 2ª ordem ou todos os requisitos normativos.
- O coeficiente de G favorável deve ser confirmado na edição normativa efetivamente adotada.
- A equivalência de Hogeslag para cargas em múltiplos níveis e fundações com molas deve ser
  justificada pela referência bibliográfica da disciplina.
- Validar os resultados com um modelo independente antes de utilizá-los como resultados finais.
