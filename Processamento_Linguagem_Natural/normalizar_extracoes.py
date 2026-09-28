# IMPORTAÇÃO DAS BIBLIOTECAS E MÓDULOS
 
from __future__ import annotations
import argparse
import json
import math
import re
import unicodedata
from pathlib import Path
from typing import Any
import pandas as pd



# COMPONENTES QUÍMICOS

COMPONENTES = [
    "FA", "MA", "Cs", "Rb", "K", "Na",
    "Pb", "Sn", "Ge",
    "I", "Br", "Cl", "F",
]


COLUNAS = [
    "i",
    "extracao_id",
    "composicao_original",

    # Sítio A
    "FA",
    "MA",
    "Cs",
    "Rb",
    "K",
    "Na",

    # Sítio B
    "Pb",
    "Sn",
    "Ge",

    # Sítio X
    "I",
    "Br",
    "Cl",
    "F",

    # Outras informações
    "eficiencia",
    "eficiencia_min",
    "eficiencia_max",
    "unidade",
    "metodo_de_deposicao",
    "arquitetura",
    "evidencia",
]


# NORMALIZAÇÃO DE TEXTO QUÍMICO

def normalizar_formula(formula):

    componentes = [
        "FA", "MA", "Cs", "Rb", "Na",
        "Pb", "Sn", "Ge",
        "Br", "Cl",
        "I", "F", "K"
    ]

    resultado = {
        c: None for c in componentes
    }

    if formula is None:
        return resultado

    s = str(formula).strip()

    if not s:
        return resultado


    # NORMALIZAÇÃO

    s = re.sub(r"\s+", "", s)

    # Fórmulas alternativas
    s = s.replace("CH3NH3", "MA")
    s = s.replace("CH(NH2)2", "FA")

    # LaTeX
    s = re.sub(r"\$_\{([^}]+)\}", r"\1", s)
    s = re.sub(r"_\{([^}]+)\}", r"\1", s)
    s = re.sub(r"_([0-9]+)", r"\1", s)

    # Subscritos Unicode
    tabela = str.maketrans(
        "₀₁₂₃₄₅₆₇₈₉",
        "0123456789"
    )

    s = s.translate(tabela)


    # TOKENS

    tokens = [
        "CH3NH3",
        "CH(NH2)2",

        "MA",
        "FA",

        "Cs",
        "Rb",
        "Na",

        "Pb",
        "Sn",
        "Ge",

        "Br",
        "Cl",

        "I",
        "F",
        "K",
    ]

    tokens = sorted(
        tokens,
        key=len,
        reverse=True
    )

    # Mapeamento
    mapa = {
        "CH3NH3": "MA",
        "CH(NH2)2": "FA",
        "MA": "MA",
        "FA": "FA",
        "Cs": "Cs",
        "Rb": "Rb",
        "Na": "Na",
        "Pb": "Pb",
        "Sn": "Sn",
        "Ge": "Ge",
        "Br": "Br",
        "Cl": "Cl",
        "I": "I",
        "F": "F",
        "K": "K",
    }


    # 3. LOCALIZAR TOKENS

    pos = 0

    while pos < len(s):

        encontrado = False

        for token in tokens:

            if not s.startswith(token, pos):
                continue

            # Evitar falsos positivos
            # F em FA
            # I em alguma sequência maior

            if token in {"F", "I", "K"}:

                # Se a posição atual começa um token maior,
                # não considerar este token.
                if any(
                    s.startswith(
                        t,
                        pos
                    )
                    for t in tokens
                    if len(t) > len(token)
                ):
                    continue

            encontrado = True

            componente = mapa[token]

            inicio = pos + len(token)

            # Encontrar o fim do coeficiente

            restante = s[inicio:]

            # CASO 1 — expressão entre parênteses       
            # I(x)
            # I(1-x)
            # Br(x)
            # Pb(1-x)

            if restante.startswith("("):

                nivel = 0
                fim = None

                for j, caractere in enumerate(
                    restante
                ):

                    if caractere == "(":
                        nivel += 1

                    elif caractere == ")":
                        nivel -= 1

                        if nivel == 0:
                            fim = j
                            break

                if fim is not None:

                    expressao = restante[
                        1:fim
                    ]

                    # Se possui variável,
                    # não podemos determinar numericamente.
                    if re.search(
                        r"[xyz]",
                        expressao
                    ):

                        resultado[
                            componente
                        ] = None

                    else:

                        try:

                            valor = float(
                                expressao
                            )

                            resultado[
                                componente
                            ] = valor

                        except ValueError:

                            resultado[
                                componente
                            ] = None

                    pos = inicio + fim + 1

                    break


            # CASO 2 — variável diretamente
            # Ix
            # Brx
            # Snx

            if restante.startswith(
                ("x", "y", "z")
            ):

                resultado[
                    componente
                ] = None

                pos = inicio + 1

                break

            
            # CASO 3 — número
            #
            # I3
            # Pb0.5
            # Br0.2

            m = re.match(
                r"(\d+(?:\.\d+)?)",
                restante
            )

            if m:

                valor = float(
                    m.group(1)
                )

                resultado[
                    componente
                ] = valor

                pos = (
                    inicio
                    + len(m.group(1))
                )

                break

            
            # CASO 4 — sem coeficiente
            #
            # MA
            # Pb
            # Cs
            # Sn

            resultado[
                componente
            ] = 1.0

            pos = inicio

            break

    
        # Nenhum token encontrado

        if not encontrado:
            pos += 1

    return resultado


# CARREGAR JSONL

def carregar_jsonl(caminho: Path):

    registros = []

    with caminho.open(
        "r",
        encoding="utf-8"
    ) as f:

        for n, linha in enumerate(f, 1):

            linha = linha.strip()

            if not linha:
                continue

            try:
                registros.append(
                    json.loads(linha)
                )

            except json.JSONDecodeError as exc:

                print(
                    f"[AVISO] Linha {n} ignorada: {exc}"
                )

    return registros



# CONVERSÃO NUMÉRICA

def numero(valor: Any):

    if valor is None or valor == "":
        return None

    if isinstance(valor, (int, float)):

        if (
            isinstance(valor, float)
            and math.isnan(valor)
        ):
            return None

        return float(valor)

    try:

        return float(
            str(valor)
            .strip()
            .replace(",", ".")
        )

    except ValueError:

        return None



# CONSTRUIR DATAFRAME

def construir_dataframe(registros):

    linhas = []

    for registro in registros:

        i = registro.get("i")

        if (
            "erro" in registro
            and "extracoes" not in registro
        ):

            print(
                f"[AVISO] i={i}: registro com erro -> "
                f"{registro['erro']}"
            )

            continue

        extracoes = registro.get(
            "extracoes"
        ) or []

        if not isinstance(
            extracoes,
            list
        ):
            continue

        for j, e in enumerate(extracoes):

            if not isinstance(
                e,
                dict
            ):
                continue

            composicao = e.get(
                "composicao_original",
                e.get("composicao")
            )

            metodo = e.get(
                "metodo_de_deposicao",
                e.get("metodo de deposicao")
            )

            linha = {

                "i": i,

                "extracao_id": j,

                "composicao_original":
                    composicao,

                **normalizar_formula(
                    composicao
                ),

                "eficiencia":
                    numero(
                        e.get("eficiencia")
                    ),

                "eficiencia_min":
                    numero(
                        e.get("eficiencia_min")
                    ),

                "eficiencia_max":
                    numero(
                        e.get("eficiencia_max")
                    ),

                "unidade":
                    e.get("unidade"),

                "metodo_de_deposicao":
                    metodo,

                "arquitetura":
                    e.get("arquitetura"),

                "evidencia":
                    e.get("evidencia"),
            }

            linhas.append(linha)

    if not linhas:

        return pd.DataFrame(
            columns=COLUNAS
        )

    df = pd.DataFrame(
        linhas
    )

    for coluna in COLUNAS:

        if coluna not in df.columns:

            df[coluna] = None

    return df[COLUNAS]



# MAIN

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--entrada",
        type=Path,
        default=Path(
            "extracoes.jsonl"
        )
    )

    parser.add_argument(
        "--saida",
        type=Path,
        default=Path(
            "tabela_perovskitas.csv"
        )
    )

    args = parser.parse_args()

    if not args.entrada.exists():

        raise FileNotFoundError(
            f"Arquivo não encontrado: "
            f"{args.entrada}"
        )

    registros = carregar_jsonl(
        args.entrada
    )

    df = construir_dataframe(
        registros
    )

    df.to_csv(
        args.saida,
        index=False,
        encoding="utf-8-sig"
    )

    print(
        f"Registros JSONL lidos: "
        f"{len(registros)}"
    )

    print(
        f"Linhas de extração geradas: "
        f"{len(df)}"
    )

    print(
        f"Arquivo salvo em: "
        f"{args.saida}"
    )


if __name__ == "__main__":
    main()

















