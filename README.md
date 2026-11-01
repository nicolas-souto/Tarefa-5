# Tarefa 5 — Estabilidade Global de Estruturas Pré-Moldadas

Aplicação acadêmica desenvolvida em Python e Streamlit para análise da estabilidade global de estruturas de concreto pré-moldado, com comparação dos procedimentos de Hogeslag e do coeficiente γz.

Atividade da disciplina de Estruturas de Concreto Pré-Moldado e Mistas, do curso de pós-graduação em Estruturas e Materiais do PPGCAM/UFPB.

Versão do programa: 1.3.

> **Atenção:** esta ferramenta tem finalidade acadêmica. Os resultados devem ser validados com um modelo independente antes de serem utilizados como resultados finais. O programa não substitui o dimensionamento estrutural nem a verificação completa dos requisitos normativos.

## Sobre a tarefa

A tarefa consiste em estudar a estabilidade global de pórticos planos de concreto pré-moldado, comparando a influência do número de pavimentos, das condições de fundação e dos cenários de carregamento na resposta estrutural.

O modelo considera quatro pilares contínuos e três vãos de 18 m por nível. As ligações viga-pilar são idealizadas como articuladas, com compatibilidade da translação horizontal dos pilares em cada nível, sem acoplamento de suas rotações.

São comparadas as estruturas E1, E2 e E3, correspondentes a um, dois e três pavimentos, e as condições de fundação F1 e F2:

- F1: bases engastadas.
- F2: bases com restrição rotacional representada por molas.

A análise contempla combinações de ações, distribuições de vento, distribuições parciais de sobrecarga e imperfeições geométricas globais.

O programa permite comparar os indicadores de estabilidade, consultar deslocamentos e esforços, identificar cenários governantes e gerar uma memória de cálculo em PDF.

## Objetivos

- Avaliar a influência da altura da estrutura e da flexibilidade das fundações na estabilidade global.
- Comparar os resultados dos procedimentos de Hogeslag e do coeficiente γz.
- Analisar a resposta de primeira ordem e de segunda ordem geométrica linearizada.
- Identificar cenários regulares, quase críticos e instáveis.
- Construir envoltórias de momentos e identificar os cenários responsáveis pelos extremos.
- Realizar uma triagem preliminar de seções quanto à estabilidade do modelo.
- Documentar entradas, hipóteses, resultados e limitações em uma memória de cálculo.

## Funcionalidades

| Recurso                     | Descrição                                                                                                      |
| --------------------------- | -------------------------------------------------------------------------------------------------------------- |
| Análise completa            | Avalia as configurações E1–E3 e F1–F2, com as combinações e os cenários de carregamento previstos no programa. |
| Análise personalizada       | Permite selecionar configurações e ajustar parâmetros de geometria, materiais e carregamento.                  |
| Indicadores de estabilidade | Apresenta γz, resultados de Hogeslag e referência modal por autovalores.                                       |
| Resultados estruturais      | Exibe deslocamentos, reações, forças normais e momentos fletores.                                              |
| Envoltórias                 | Reúne os extremos dos cenários analisados e identifica seus casos governantes.                                 |
| Diagnóstico numérico        | Separa os casos quase críticos das envoltórias regulares de segunda ordem.                                     |
| Triagem de seções           | Registra as seções testadas e sua classificação quanto à estabilidade.                                         |
| Exportação                  | Gera memória de cálculo em PDF e tabelas em CSV.                                                               |

## Arquivos principais

| Arquivo                         | Finalidade                                     |
| ------------------------------- | ---------------------------------------------- |
| `tarefa_5_hogeslag_1_3.py`      | Código principal e interface Streamlit.        |
| `requirements_hogeslag_1_3.txt` | Dependências Python.                           |
| `README.md`                     | Apresentação da atividade e instruções de uso. |

## Como rodar

As instruções abaixo consideram Windows com PowerShell e Python 3.13.

### 1. Instale o Python

Baixe e instale o Python 3.13:

https://www.python.org/downloads/

No instalador do Windows, marque a opção `Add Python to PATH`. Após a instalação, abra uma nova janela do PowerShell.

Verifique a instalação:

```powershell
py -3.13 --version
```

### 2. Acesse a pasta do projeto

Baixe o repositório e abra o PowerShell na pasta que contém os arquivos:

```text
Tarefa-5/
├── tarefa_5_hogeslag_1_3.py
├── requirements.txt
└── README.md
```

Se necessário, navegue até a pasta:

```powershell
cd "C:\caminho\para\Tarefa-5"
```

Substitua o caminho pelo local em que você salvou o projeto.

### 3. Crie um ambiente virtual

```powershell
py -3.13 -m venv .venv
```

Os próximos comandos utilizam diretamente o Python do ambiente virtual, sem precisar ativá-lo no PowerShell.

### 4. Instale as dependências

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

O arquivo de requisitos contém:

```text
streamlit>=1.35
numpy>=1.26
pandas>=2.0
matplotlib>=3.8
reportlab>=4.0
```

### 5. Inicie a aplicação

```powershell
.\.venv\Scripts\python.exe -m streamlit run tarefa_5_hogeslag_1_3.py
```

Acesse o endereço local informado no terminal.

Para encerrar o servidor, pressione `Ctrl+C` no PowerShell.

> Execute os comandos na pasta do projeto. Se o script tiver outro nome, ajuste o comando de inicialização para corresponder ao arquivo existente.

## Como executar uma análise

1. Selecione o escopo: análise completa do exercício ou análise personalizada.
2. Configure os parâmetros de geometria, materiais, carregamento e rigidez.
3. Informe um desaprumo global maior que zero e registre sua origem ou critério de cálculo.
4. Confira as equivalências adotadas no procedimento de Hogeslag.
5. Se habilitar a verificação complementar de G favorável, informe o coeficiente e sua fonte normativa.
6. Se desejar, habilite a busca preliminar de seção e defina o intervalo de teste.
7. Clique em `Executar análise`.
8. Consulte os resultados, os casos governantes e os diagnósticos de estabilidade.
9. Exporte as tabelas e a memória de cálculo.

O programa não determina automaticamente os valores normativos que devem ser informados pelo usuário.

## Recursos e alterações da versão 1.3

### Envoltórias de segunda ordem

As envoltórias regulares de segunda ordem excluem os cenários classificados como quase críticos.

Esses cenários são armazenados e apresentados separadamente, apenas para diagnóstico de sensibilidade numérica.

### Seleção do caso detalhado

O caso detalhado pode ser escolhido pelos seguintes critérios:

- Caso representativo: mediana de `gamma_z` entre os casos regulares.
- Maior `gamma_z` finito.
- Caso mais instável: menor autovalor normalizado.
- Maior módulo de momento válido de segunda ordem.

### Desaprumo global

A interface exige:

- Desaprumo global maior que zero.
- Indicação da origem ou do critério adotado.

O programa não escolhe automaticamente o valor normativo.

### Verificação complementar de G favorável

Exige que o usuário informe o coeficiente adotado e a fonte normativa correspondente.

A ação variável Q é retirada dessa verificação complementar.

### Busca progressiva de seção

Cada seção testada é registrada.

Uma seção só é aceita na triagem quando atende simultaneamente aos critérios:

- Matriz positiva definida.
- Não classificação como quase crítica.

### Memória de cálculo em PDF

Inclui:

- Dados de entrada e hipóteses.
- Resultados e casos governantes.
- Contagem de cenários regulares, quase críticos e instáveis.
- Critério de seleção do caso detalhado.
- Conclusão automática.
- Limitações e validações pendentes.

## Hipóteses e limitações

### Uso do coeficiente γz

O coeficiente γz é apresentado para comparação didática. Como as estruturas E1, E2 e E3 possuem menos de quatro pavimentos, sua interpretação deve respeitar a ressalva do enunciado e a aplicabilidade do procedimento adotado.

### Modelo de segunda ordem

A análise `K_E - K_G` corresponde à segunda ordem geométrica linearizada, com forças normais de primeira ordem mantidas constantes.

Não se trata de uma análise não linear incremental completa.

### Critério de quase criticidade

O limite numérico de quase criticidade é um critério computacional, não um limite normativo.

Os resultados desses cenários são apresentados como diagnóstico e não devem ser usados como envoltórias regulares ou esforços de dimensionamento.

### Triagem de seção

A busca preliminar de seção não constitui dimensionamento e não verifica:

- Resistência.
- Armaduras.
- Esbeltez local.
- Efeitos locais de segunda ordem.
- Todos os requisitos normativos.

A aceitação de uma seção na triagem refere-se apenas aos critérios de estabilidade avaliados no cenário considerado.

### Ações permanentes favoráveis

O coeficiente de G favorável deve ser confirmado na edição normativa efetivamente adotada.

O usuário é responsável por informar o coeficiente e sua fundamentação.

### Equivalências de Hogeslag

As equivalências adotadas para cargas em múltiplos níveis e fundações com molas devem ser justificadas pela referência bibliográfica da disciplina.

Os resultados dependem dessas hipóteses e não constituem, por si só, uma validação independente do procedimento.

## Validação dos resultados

Antes de utilizar os resultados na entrega final:

- Confira dados de entrada, unidades e convenções de sinais.
- Revise combinações de ações e distribuições de carregamento.
- Verifique a fundamentação do desaprumo e dos coeficientes informados.
- Examine os alertas de quase criticidade e instabilidade.
- Confira os resíduos de equilíbrio e de recuperação dos diagramas.
- Compare os resultados com um modelo independente.
- Registre no relatório as hipóteses, simplificações e limitações adotadas.
