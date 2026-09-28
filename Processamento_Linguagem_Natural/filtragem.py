# IMPORTAÇÃO DAS BIBLIOTECAS E MÓDULOS

import pandas as pd
from chemdataextractor.doc import Document, Paragraph


# IMPORTAÇÃO DOS DADOS

caminho = 'dados/PLNI_data.csv'
dados = pd.read_csv(caminho)

COL_TITULO   = 'Article Title'
COL_ABSTRACT = 'Abstract'
COL_ANO      = 'Publication Year'
COL_KEYWORDS = 'Author Keywords'
COLUNAS = [c for c in [COL_TITULO, COL_ABSTRACT, COL_ANO, COL_KEYWORDS] if c in dados.columns]

df = (dados[COLUNAS]
      .dropna(subset=[COL_ABSTRACT])     # sem resumo, não dá pra comparar nada
      .drop_duplicates(subset=[COL_ABSTRACT])  # remove artigos duplicados no export
      .reset_index(drop=True))

print(f'Artigos utilizáveis: {len(df)} (de {len(dados)} originais)')



# FILTRAGEM COM REGEX + CHEMDATAEXTRACTOR 

def processar_dataframe(df, coluna_texto="abstract"):
    def extrair(texto):
        if not isinstance(texto, str) or not texto.strip():
            return []
        doc = Document(Paragraph(texto))
        return [cem.text for cem in doc.cems]
 
    df = df.copy()
    df["entidades_quimicas_cde"] = df[coluna_texto].apply(extrair)
    return df

df_filtrado = processar_dataframe(df, COL_ABSTRACT)

df_filtrado.to_csv("dados_filtrados.csv")

print(df_filtrado)