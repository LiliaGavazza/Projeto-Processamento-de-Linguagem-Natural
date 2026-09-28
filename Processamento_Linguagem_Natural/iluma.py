# IMPORTAÇÃO DAS BIBLIOTECAS E MÓDULOS

import os
import json
import re
import time
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import pandas as pd
from openai import OpenAI


# CONFIGURAÇÕES

BASE_URL = "https://iluma.cnpem.br:4000/v1"
MODELO = "iluma"
TEMP = 0.5
TIMEOUT = 600.0
MAX_RETRIES = 0
ARQUIVO_ENTRADA = Path("dados_filtrados_formulas2.csv")
SAIDA = Path("extracoes.jsonl")

# Para o primeiro teste, deixe pequeno.
# Depois você pode colocar None para processar todos.
'''LIMITE = 30'''
LIMITE = None

# Número de chamadas simultâneas à ILUMA.
WORKERS = 3

# A conferência do primeiro candidato faz uma chamada extra.
# Deixe False durante o processamento normal.
TESTAR_EXEMPLO = False


# Função de Log
def log(mensagem=""):
    print(mensagem, flush=True)


# INÍCIO

log("=" * 70)
log("INÍCIO DO PROCESSAMENTO ILUMA")
log("=" * 70)
log(f"HOST: {os.uname().nodename}")
log(f"PYTHON: {sys.executable}")
log(f"PYTHON VERSION: {sys.version.split()[0]}")
log(f"DIRETÓRIO: {Path.cwd()}")
log(f"DATA/HORA: {time.strftime('%Y-%m-%d %H:%M:%S')}")
log("=" * 70)

log("[1/7] Verificando API_TOKEN...")
API_TOKEN = os.environ.get("API_TOKEN")
if not API_TOKEN:
    log("ERRO: API_TOKEN não está definida.")
    sys.exit(1)
log("API_TOKEN encontrada.")

cliente = OpenAI(
    base_url=BASE_URL,
    api_key=API_TOKEN,
    timeout=TIMEOUT,
    max_retries=MAX_RETRIES,
)


# FUNÇÃO DE CHAMADA À API

def perguntar(mensagens, **kw):
    kw.setdefault("temperature", TEMP)
    inicio = time.time()

    try:
        resposta = cliente.chat.completions.create(
            model=MODELO,
            messages=mensagens,
            **kw
        )
        tempo = time.time() - inicio
        log(f"      Resposta recebida em {tempo:.1f} s")
        return resposta.choices[0].message.content

    except Exception as e:
        tempo = time.time() - inicio
        log(
            f"      ERRO após {tempo:.1f} s: "
            f"{type(e).__name__}: {e}"
        )
        raise


# TESTE DA API

log("[2/7] Testando conexão com a ILUMA...")

try:
    teste = perguntar([
        {
            "role": "user",
            "content": "Responda apenas: conexão OK."
        }
    ])
    log(f"      Resposta de teste: {teste}")

except Exception as e:
    log("")
    log("=" * 70)
    log("FALHA NO TESTE DA API")
    log("=" * 70)
    log(f"Tipo: {type(e).__name__}")
    log(f"Erro: {e}")
    log("")
    log("O programa NÃO continuará para os artigos.")
    log("Verifique o API_TOKEN e o acesso à ILUMA.")
    log("=" * 70)
    sys.exit(1)


# Leitura do CSV
log("[3/7] Lendo arquivo CSV...")

if not ARQUIVO_ENTRADA.exists():
    log(f"ERRO: arquivo não encontrado: {ARQUIVO_ENTRADA}")
    sys.exit(1)

dados = pd.read_csv(ARQUIVO_ENTRADA)

log(f"      Arquivo: {ARQUIVO_ENTRADA}")
log(f"      Linhas: {len(dados)}")
log(f"      Colunas: {len(dados.columns)}")

if "Abstract" not in dados.columns:
    log("ERRO: coluna 'Abstract' não encontrada.")
    log(f"Colunas disponíveis: {list(dados.columns)}")
    sys.exit(1)

abstracts = (
    dados["Abstract"]
    .dropna()
    .astype(str)
    .tolist()
)
log(f"      Resumos válidos: {len(abstracts)}")


# FILTROS

log("[4/7] Aplicando filtros aos resumos...")

CHAVE = re.compile(
    r"efficiency",
    re.IGNORECASE
)

VALOR = re.compile(
    r"\d+(?:[.,]\d+)?%"
)
A_FORMULA = r"""
(?: MA | FA | Cs | Rb | Na | K | CH3NH3 | CH\(NH2\)2 )
"""
B_FORMULA = r"""
(?: Pb | Sn | Ge )
"""
X_FORMULA = r"""
(?: I | Br | Cl | F )
"""
STOICH = r"""
(?:
    \d+(?:\.\d+)?
    | x
    | y
    | z
    | \d+(?:\.\d+)?-x
    | \d+(?:\.\d+)?-y
    | \d+(?:\.\d+)?-z
)?
"""

FORMULA_ABX = re.compile(
    rf"""
    (?<![A-Za-z0-9])
    (?P<A>{A_FORMULA})
    (?P<A_st>{STOICH})
    (?P<B>{B_FORMULA})
    (?P<B_st>{STOICH})
    (?P<X>{X_FORMULA})
    (?P<X_st>{STOICH})
    (?![A-Za-z0-9])
    """,
    re.IGNORECASE | re.VERBOSE
)

def vale_a_chamada(texto: str) -> bool:
    return bool(
        CHAVE.search(texto)
        and VALOR.search(texto)
        and FORMULA_ABX.search(texto)
    )

candidatos = [
    a for a in abstracts
    if vale_a_chamada(a)
]

if abstracts:
    porcentagem = (
        len(candidatos) / len(abstracts)
    )

else:
    porcentagem = 0

log(f"      Resumos totais: {len(abstracts)}")
log(f"      Candidatos: {len(candidatos)}")
log(f"      Percentual: {porcentagem:.1%}")
log(
    f"      Chamadas economizadas: "
    f"{len(abstracts) - len(candidatos)}"
)

if not candidatos:
    log("")
    log("Nenhum candidato encontrado.")
    log("O programa será encerrado.")
    sys.exit(0)



# PARSER JSON
def ler_json(texto: str) -> dict:
    # Transformar a resposta da ILUMA em JSON.
    if not texto:
        return {}
    t = texto.strip()

    # Remove markdown ```json ... ```
    if t.startswith("```"):
        t = re.sub(
            r"^```(?:json)?\s*",
            "",
            t,
            flags=re.IGNORECASE
        )
        t = re.sub(
            r"\s*```$",
            "",
            t
        )

    # Primeira tentativa
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass

    # Segunda tentativa: procurar trecho entre { }
    m = re.search(
        r"\{.*\}",
        t,
        re.S
    )
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    log(
        "      AVISO: resposta não pôde ser interpretada "
        "como JSON."
    )
    log(f"      Resposta: {t[:300]}")
    return {}


# PROMPT

SISTEMA = """Você é um modelo especializado em extração de informações científicas de resumos de artigos sobre células solares de perovskita.
Sua tarefa é ler o resumo fornecido e extrair informações sobre células solares de perovskita que possam ser utilizadas posteriormente para analisar a relação entre composição e eficiência.
Responda APENAS com JSON válido. Não inclua explicações, comentários ou cercas de Markdown.

INFORMAÇÕES A EXTRAIR
Para cada célula ou composição de perovskita associada a uma eficiência explicitamente informada no resumo, extraia:

* composição química original da perovskita;
* eficiência da célula;
* método de deposição da perovskita, se informado;
* arquitetura da célula, se informada;
* uma evidência textual curta que justifique a extração.

1. COMPOSIÇÃO QUÍMICA

O campo "composicao_original" deve conter a fórmula ou composição química da perovskita exatamente como aparece no resumo.

Exemplos de composições:

* MAPbI3
* FAPbI3
* CsPbI3
* FA0.8MA0.2PbI3
* Cs0.1FA0.9PbI3
* MAPb0.5Sn0.5I3
* FA0.8MA0.2Pb(I0.9Br0.1)3

Preserve a composição original e sua notação tanto quanto possível.
NÃO use como "composicao_original" descrições genéricas ou arquiteturas, como:

* "perovskite solar cell"
* "perovskite"
* "tandem solar cell"
* "4T tandem solar cell"
* "perovskite/silicon tandem"
* "quantum dot solar cell"
* "all-solid-state perovskite"

Se o resumo informar uma eficiência, mas não fornecer uma composição química específica, use:

"composicao_original": null

Não invente uma composição com base em conhecimento externo.

2. EFICIÊNCIA

Extraia somente valores que o resumo identifique explicitamente como eficiência da célula, especialmente power conversion efficiency (PCE).
Registre o valor numérico sem o símbolo "%".

Exemplo:

Se o texto disser:
"the solar cell achieved a power conversion efficiency of 21.4%"

retorne:

"eficiencia": 21.4

Não confunda eficiência com:

* band gap;
* fill factor;
* open-circuit voltage;
* short-circuit current density;
* aumento percentual;
* redução percentual;
* diferença percentual;
* ganho relativo;
* qualquer outra propriedade.

Se um percentual representar apenas uma comparação, não o registre como eficiência.

Exemplo:

"the efficiency increased by 20%"
não significa que a eficiência seja 20%.

Nesse caso, procure no texto o valor absoluto da eficiência. Se ele não estiver presente, não faça uma inferência.

3. MÚLTIPLAS EFICIÊNCIAS

Se o resumo apresentar diferentes células ou composições com diferentes eficiências, crie uma entrada separada para cada combinação que possa ser associada de forma confiável.
Não agrupe diferentes eficiências em uma única string.

INCORRETO:

"eficiencia": "9.7 and 10.9%"

CORRETO:

criar entradas separadas quando for possível determinar qual eficiência pertence a qual célula.

Se não for possível determinar a associação entre uma composição e uma eficiência, não invente a associação.

4. INTERVALOS

Se uma eficiência for apresentada como intervalo, preserve o intervalo usando dois campos:

"eficiencia_min": 15.0,
"eficiencia_max": 16.0

Não calcule a média do intervalo.
Para uma eficiência única, use:

"eficiencia": 21.4,
"eficiencia_min": null,
"eficiencia_max": null

5. MÉTODO DE DEPOSIÇÃO

Extraia o método de deposição da camada de perovskita quando explicitamente informado.
Exemplos:

* spin coating
* blade coating
* slot-die coating
* thermal evaporation
* vapor deposition
* electrodeposition
* antisolvent method

Não invente o método. Se não estiver informado:
"metodo_de_deposicao": null
IMPORTANTE: não confunda tratamento, processamento ou fabricação de outras camadas da célula com o método de deposição da perovskita.

6. ARQUITETURA

Extraia a arquitetura da célula quando explicitamente informada.
Exemplos:
* n-i-p
* p-i-n
* conventional
* inverted
* 4T tandem
* perovskite/silicon tandem
Não confunda arquitetura com composição química.
Se não estiver informada:
"arquitetura": null

7. EVIDÊNCIA

Para cada extração, forneça uma pequena evidência textual retirada do resumo.
A evidência deve conter, sempre que possível, a composição e a eficiência utilizadas na extração.
Não invente ou reescreva informações que não estejam no resumo.

8. INFORMAÇÃO AUSENTE

Não invente informações.
Se uma informação não estiver explicitamente presente no resumo, use null.
É preferível retornar null a inferir uma informação.

FORMATO DE SAÍDA

Use exatamente este formato:
{
"extracoes": [
{
"composicao_original": "FA0.8MA0.2PbI3",
"eficiencia": 21.4,
"eficiencia_min": null,
"eficiencia_max": null,
"unidade": "%",
"metodo_de_deposicao": "spin coating",
"arquitetura": "n-i-p",
"evidencia": "FA0.8MA0.2PbI3 solar cells achieved a power conversion efficiency of 21.4%."
}
]
}

Se não houver uma extração válida, responda:

{
"extracoes": []
}

Não inclua nenhum texto fora do JSON.
"""

EXEMPLO_ENTRADA = (
    "Precise control of the spin coating time made the average power "
    "conversion efficiency of the MAPbI3 solar cells increase from "
    "12–13% to 15–16%."
)

EXEMPLO_SAIDA = json.dumps(
    {
          "extracoes": [{
        "composicao_original": "MAPbI3",
        "eficiencia": None,
        "eficiencia_min": 15.0,
        "eficiencia_max": 16.0,
        "unidade": "%",
        "metodo_de_deposicao": "spin coating",
        "arquitetura": None,
        "evidencia": (
            "the average power conversion efficiency of the MAPbI3 "
            "solar cells increase ... to 15–16%"
        )
    }]
    },
    ensure_ascii=False
)


# EXTRAÇÃO

def extrair(resumo: str) -> list[dict]:
    bruto = perguntar(
        [
            {
                "role": "system",
                "content": SISTEMA
            },
            {
                "role": "user",
                "content": EXEMPLO_ENTRADA
            },
            {
                "role": "assistant",
                "content": EXEMPLO_SAIDA
            },
            {
                "role": "user",
                "content": resumo
            },
        ]
    )

    resultado = ler_json(bruto)
    extracoes = resultado.get(
        "extracoes",
        []
    )

    if not isinstance(extracoes, list):
        log(
            "      AVISO: campo 'extracoes' "
            "não é uma lista."
        )
        return []
    return extracoes


# PROCESSAMENTO DOS ARTIGOS

def processar_item(item):
    # Processa um único abstract.
    # Esta função é executada pelos workers em paralelo.
    # Ela não escreve diretamente no JSONL; o thread principal faz isso
    # quando cada Future termina, evitando escrita concorrente no arquivo.
    i, texto = item
    inicio = time.time()

    log("")
    log(f"[{i}] Iniciando processamento")
    log(f"      Tamanho do resumo: {len(texto)} caracteres")

    try:
        log(f"      Enviando requisição para a ILUMA...")
        resultado = extrair(texto)
        tempo = time.time() - inicio

        linha = {
            "i": i,
            "extracoes": resultado
        }

        log(
            f"[{i}] Concluído em {tempo:.1f} s "
            f"({len(resultado)} extrações)"
        )
        return linha, tempo

    except Exception as e:
        tempo = time.time() - inicio

        linha = {
            "i": i,
            "erro": f"{type(e).__name__}: {e}"
        }

        log(
            f"[{i}] ✗ Falhou em {tempo:.1f} s: "
            f"{type(e).__name__}: {e}"
        )

        return linha, tempo


def rodar_lote(
    textos,
    comeco=0,
    limite=None,
    workers=WORKERS
):
    #cProcessa abstracts em paralelo e salva cada resultado no JSONL.
    # `workers` controla quantas chamadas à ILUMA podem ocorrer ao mesmo tempo.
    # Resultados são gravados assim que cada chamada termina.
    # Um item só é considerado concluído se a linha existente tiver
    # o campo `extracoes`. Linhas com `erro` serão tentadas novamente
    # numa próxima execução.

    log("")
    log("=" * 70)
    log("[6/7] PROCESSANDO ARTIGOS")
    log("=" * 70)
    log(f"      Chamadas simultâneas: {workers}")


    # Descobrir o que já foi processado com sucesso

    feitos = set()
    if SAIDA.exists():
        log(f"      Encontrado arquivo existente: {SAIDA}")

        with SAIDA.open(
            "r",
            encoding="utf-8"
        ) as f:
            for numero_linha, linha in enumerate(f, start=1):
                if not linha.strip():
                    continue

                try:
                    d = json.loads(linha)
                except json.JSONDecodeError:
                    log(
                        f"      AVISO: linha {numero_linha} "
                        "do JSONL está inválida e será ignorada."
                    )
                    continue

                # Se houve erro, não marca como feito.
                # Assim o item pode ser reprocessado numa próxima execução.
                if "i" in d and "extracoes" in d:
                    feitos.add(d["i"])

        log(f"      Itens já processados com sucesso: {len(feitos)}")


    # Definir artigos que serão processados
    
    alvos = list(enumerate(textos))[comeco:]

    # Primeiro remove os que já foram concluídos.
    alvos = [
        item for item in alvos
        if item[0] not in feitos
    ]

    if limite is not None:
        alvos = alvos[:limite]

    total = len(alvos)
    log(f"      Itens nesta execução: {total}")

    if total == 0:
        log("      Nenhum item novo para processar.")
        return


    # Processamento paralelo

    inicio_total = time.time()
    concluidos = 0

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futuros = {
            executor.submit(processar_item, item): item[0]
            for item in alvos
        }

        with SAIDA.open(
            "a",
            encoding="utf-8"
        ) as f:
            for futuro in as_completed(futuros):
                i = futuros[futuro]

                try:
                    linha, tempo = futuro.result()
                except Exception as e:
                    # Proteção extra: processar_item já captura exceções,
                    # mas mantemos este bloco para não derrubar o lote inteiro.
                    linha = {
                        "i": i,
                        "erro": f"{type(e).__name__}: {e}"
                    }
                    tempo = None

                # Salva imediatamente quando o Future termina.
                f.write(
                    json.dumps(
                        linha,
                        ensure_ascii=False
                    ) + "\n"
                )
                f.flush()

                concluidos += 1

                log(
                    f"      Progresso: {concluidos}/{total} "
                    f"({100 * concluidos / total:.1f}%)"
                )
                log(f"      Resultado do índice {i} salvo em {SAIDA}")

    tempo_total = time.time() - inicio_total

    log("")
    log("=" * 70)
    log("PROCESSAMENTO CONCLUÍDO")
    log("=" * 70)
    log(f"      Itens processados: {concluidos}")
    log(f"      Tempo total: {tempo_total:.1f} s ({tempo_total / 60:.1f} min)")
    if concluidos:
        log(f"      Tempo médio por item: {tempo_total / concluidos:.1f} s")


# ESTIMATIVA DE TOKENS

def estimar(
    textos,
    sistema=SISTEMA,
    exemplo=EXEMPLO_ENTRADA + EXEMPLO_SAIDA
):
    if not textos:
        return 0, 0

    fixo = (
        len(sistema)
        + len(exemplo)
    )

    chars = (
        sum(len(t) for t in textos)
        + fixo * len(textos)
    )

    tokens = chars / 4

    return len(textos), tokens



# CONFERÊNCIA DAS EXTRAÇÕES

def confere_no_texto(
    extracoes: list[dict],
    origem: str
) -> list[dict]:

    # Faz uma verificação simples para saber se a eficiência
    # retornada aparece no texto original.

    # Como 'eficiencia' pode vir como string ("15–16%"),
    # esta função não assume que seja número.

    for e in extracoes:
        valor = e.get("eficiencia")
        if valor is None:
            e["confere"] = None
            continue
        valor_str = str(valor)
        # Extrai números do valor retornado
        numeros = re.findall(
            r"\d+(?:[.,]\d+)?",
            valor_str
        )
        encontrados = []
        for numero in numeros:
            numero = numero.replace(
                ",",
                "."
            )
            encontrados.append(numero)

        e["confere"] = any(
            numero in origem
            or numero.replace(".", ",") in origem
            for numero in encontrados
        )
    return extracoes


# AMOSTRA PARA CONFERÊNCIA MANUAL

def amostra_para_conferir(
    caminho=SAIDA,
    n=30,
    semente=42
):
    linhas = []
    if not Path(caminho).exists():
        log(
            f"Arquivo não encontrado: {caminho}"
        )
        return pd.DataFrame()
    with Path(caminho).open(
        encoding="utf-8"
    ) as f:
        for linha in f:
            if not linha.strip():
                continue
            try:
                d = json.loads(linha)
            except json.JSONDecodeError:
                continue
            for e in d.get(
                "extracoes",
                []
            ):
                i = d.get("i")
                if i is None:
                    continue
                if i >= len(candidatos):
                    continue
                linhas.append(
                    {
                        "i": i,
                        "composicao": e.get(
                            "composicao_original",
                            e.get("composicao")
                        ),
                        "%": e.get(
                            "eficiencia"
                        ),
                        "metodo_de_deposicao":
                            e.get(
                                "metodo_de_deposicao",
                                e.get("metodo de deposicao")
                            ),
                        "resumo":
                            candidatos[i][:300]
                    }
                )

    tab = pd.DataFrame(linhas)
    log(
        f"{len(tab)} extrações no arquivo."
    )
    if tab.empty:
        return tab
    return tab.sample(
        min(n, len(tab)),
        random_state=semente
    )


# EXECUÇÃO PRINCIPAL

log("[5/7] Preparando processamento...")
log(
    f"      Limite desta execução: {LIMITE}"
)
log(
    f"      Chamadas simultâneas: {WORKERS}"
)
rodar_lote(
    candidatos,
    limite=LIMITE,
    workers=WORKERS
)


# ESTIMATIVA DE CUSTO / TOKENS

log("")
log("[7/7] Calculando estimativas...")
n, tok = estimar(candidatos)
n_tudo, tok_tudo = estimar(abstracts)
log(
    f"      Com filtro: "
    f"{n} chamadas · "
    f"~{tok / 1000:.0f}k tokens de entrada"
)
log(
    f"      Sem filtro: "
    f"{n_tudo} chamadas · "
    f"~{tok_tudo / 1000:.0f}k tokens de entrada"
)

if tok_tudo > 0:
    economia = (
        100 * (1 - tok / tok_tudo)
    )
    log(
        f"      Economia estimada: "
        f"{economia:.1f}%"
    )



# TESTE OPCIONAL DO PRIMEIRO CANDIDATO

if TESTAR_EXEMPLO:
    log("")
    log("=" * 70)
    log("CONFERÊNCIA DO PRIMEIRO CANDIDATO")
    log("=" * 70)
    if candidatos:
        exemplo = candidatos[0]
        log(
            exemplo[:500]
            + "..."
        )
        try:
            extracoes = extrair(
                exemplo
            )
            extracoes = confere_no_texto(
                extracoes,
                exemplo
            )
            for e in extracoes:
                marca = {
                    True: "OK",
                    False: "NÃO ENCONTRADO",
                    None: "SEM VALOR"
                }[
                    e.get("confere")
                ]
                log(
                    f"  {e.get('composicao')}: "
                    f"{e.get('eficiencia')} "
                    f"→ {marca}"
                )
        except Exception as e:
            log(
                f"Erro na conferência: "
                f"{type(e).__name__}: {e}"
            )
log("")
log("=" * 70)
log("FIM DO PROGRAMA")
log("=" * 70)