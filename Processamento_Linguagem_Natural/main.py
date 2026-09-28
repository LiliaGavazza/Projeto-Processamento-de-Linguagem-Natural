# IMPORTAÇÃO DAS BIBLIOTECAS E MÓDULOS

import re
import numpy as np
import pandas as pd
import spacy
from sklearn.feature_extraction.text import CountVectorizer

pd.set_option('display.max_colwidth', 120)
pd.set_option('display.width', 160)

# modelo de pipeline
nlp = spacy.load('en_core_web_sm')


# IMPORTAÇÃO DOS DADOS

CAMINHO_CSV = 'dados_filtrados_formulas2.csv'   

dados = pd.read_csv(CAMINHO_CSV)

COL_TITULO   = 'Article Title'
COL_ABSTRACT = 'Abstract'
COL_ANO      = 'Publication Year'
COL_KEYWORDS = 'Author Keywords'

COLUNAS = [c for c in [COL_TITULO, COL_ABSTRACT, COL_ANO, COL_KEYWORDS] if c in dados.columns]

df = (dados[COLUNAS]
      .dropna(subset=[COL_ABSTRACT])            # sem resumo, não dá pra comparar nada
      .drop_duplicates(subset=[COL_ABSTRACT])   # remove artigos duplicados no export
      .reset_index(drop=True))                  # reindexa 0..N-1 

corpus = df[COL_ABSTRACT].tolist()


# PROCESSAMENTO

componentes_desnecessarios = ['ner', 'parser']

# Unidades e siglas que precisam manter a caixa original mesmo não seguindo o
# padrão símbolo-de-elemento+número (ex.: 'eV' não é 'E'+'v' de elemento,
# 'PCE'/'XRD' são siglas). 
UNIDADES_SIGLAS = {
    'eV', 'meV', 'nm', 'mV', 'mA', 'cm2', 'mW', 'PCE', 'FF', 'Voc', 'Jsc',
    'XRD', 'SEM', 'TEM', 'UV', 'PL', 'TRPL', 'EQE', 'HOMO', 'LUMO',
}

# Todos os símbolos da tabela periódica 
# Usado só para reconhecer o símbolo ISOLADO (sem número
# nem outra maiúscula grudada, ex.: '(Pb)', 'lead (Pb)').
SIMBOLOS_ELEMENTOS = {
    'H','He','Li','Be','B','C','N','O','F','Ne','Na','Mg','Al','Si','P','S','Cl','Ar',
    'K','Ca','Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn','Ga','Ge','As','Se','Br','Kr',
    'Rb','Sr','Y','Zr','Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd','In','Sn','Sb','Te','I','Xe',
    'Cs','Ba','La','Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu',
    'Hf','Ta','W','Re','Os','Ir','Pt','Au','Hg','Tl','Pb','Bi','Po','At','Rn',
    'Fr','Ra','Ac','Th','Pa','U','Np','Pu','Am','Cm','Bk','Cf','Es','Fm','Md','No','Lr',
}
# Desses, os que colidem com palavra comum do inglês (minúsculo == palavra real):
# In/as/at/he/i/is-não/no/be/or-não/us-não... Excluímos da preservação "isolada"
# porque não dá pra distinguir 'In this work' (preposição) de 'In (indium)' só
# pela caixa — nesses casos o token só é preservado se vier grudado numa
# fórmula (ex.: 'InCl3'), regra já coberta pelo bloco de fórmula abaixo.
ELEMENTOS_AMBIGUOS = {'In', 'As', 'At', 'He', 'I', 'No', 'Be'}
SIMBOLOS_ISOLADOS_SEGUROS = SIMBOLOS_ELEMENTOS - ELEMENTOS_AMBIGUOS

def preservar_caixa_texto(texto):
    # Mesma decisão de preservar_caixa, mas a partir de uma STRING (não do
    # Token do spaCy) — necessário porque às vezes o spaCy funde um token com
    # a pontuação vizinha (ex.: 'eV.' sai como um token só) e só sabemos decidir
    # depois de aparar essa pontuação do texto.
    if texto in UNIDADES_SIGLAS:
        return True
    if re.match(r'^[A-Z][a-z]?', texto) and re.search(r'[A-Z0-9]', texto[1:]):
        return True
    if texto in SIMBOLOS_ISOLADOS_SEGUROS:
        return True
    if texto.isupper() and len(texto) >= 2:
        return True
    return False

def preservar_caixa(token):
    # True se o token parece ser fórmula/símbolo químico, unidade ou sigla —
    # casos em que a maiúscula carrega informação (Pb != pb) e não deve ser
    # apagada pelo lowercasing feito 'na mão' dentro das funções de tokenização.
    return preservar_caixa_texto(token.text)


# Fórmulas dopadas/mistas ('PbI3-xClx', 'MASnxPb1-xI3'...) são um problema à
# parte: o tokenizer do spaCy NÃO quebra parênteses no meio de um token (por
# isso 'Cs0.05(MA0.17FA0.83)0.95Pb(I0.83Br0.17)3' já sai inteiro), mas SEMPRE
# quebra no hífen — então 'PbI3-xClx' vira 3 tokens: 'PbI3', '-', 'xClx'.
# Checar caixa/fórmula token a token não resolve isso: precisamos primeiro
# RECOLAR os tokens que estavam colados no texto original (sem espaço entre
# eles) e só depois decidir, no texto reconstituído, se é uma fórmula.

# Pontuação de borda que cortamos de um grupo recolado (fim de frase, vírgula
# de lista...). '(' e ')' ficam de fora dessa lista de propósito: fazem parte
# da notação de fórmulas mistas.
PONTUACAO_DE_BORDA = {'.', ',', ';', ':', '!', '?', '"', "'", '``', "''"}

def agrupar_tokens_colados(doc):
    # [[tokens sem espaço entre si]] — reconstitui no doc os pedaços que o
    # tokenizer do spaCy separou (hífen, parênteses de abertura) mas que, no
    # texto original, estavam grudados um no outro.
    grupos, atual = [], []
    for t in doc:
        if t.is_space:
            continue
        atual.append(t)
        if t.whitespace_ != '':      # há espaço DEPOIS deste token -> fecha o grupo
            grupos.append(atual)
            atual = []
    if atual:
        grupos.append(atual)
    return grupos

def aparar_bordas(grupo):
    # Tira pontuação de frase (., ; : ...) das pontas do grupo, mas mantém
    # o que estiver por dentro (hífen de dopagem, parênteses de composição).
    ini, fim = 0, len(grupo)
    while ini < fim and grupo[ini].text in PONTUACAO_DE_BORDA:
        ini += 1
    while fim > ini and grupo[fim - 1].text in PONTUACAO_DE_BORDA:
        fim -= 1
    return grupo[ini:fim]

def aparar_pontuacao_textual(texto):
    # Tira pontuação de PONTUACAO_DE_BORDA das pontas de uma string 
    # Cobre o caso raro em que o spaCy funde a unidade
    # com o ponto final no mesmo token (ex.: 'eV.' vira um token só).
    while texto and texto[-1] in PONTUACAO_DE_BORDA:
        texto = texto[:-1]
    while texto and texto[0] in PONTUACAO_DE_BORDA:
        texto = texto[1:]
    return texto

def parece_formula_colada(texto):
    # Heurística para um texto recolado de VÁRIOS tokens:
    # tem letra + dígito -> é praticamente certo que é fórmula/composição
    # química (ex.: 'PbI3-xClx', 'MASnxPb1-xI3'). Palavras inglesas hifenizadas
    # coladas do mesmo jeito ('lead-free', 'high-efficiency') não têm dígito e
    # por isso não caem aqui — seguem o tratamento normal, token a token.
    return bool(re.search(r'[A-Za-z]', texto)) and bool(re.search(r'\d', texto))

def normaliza(token):
    # Preserva a caixa de fórmulas/unidades/siglas; minusculiza o resto (palavras comuns).
    return token.text if preservar_caixa(token) else token.text.lower()

def _tokenizar(texto_doc, *, somente_alpha, lematizar, remover_stop):
    # Motor comum de tok_texto / tok_alpha / tok_lema. Processa o doc em
    # grupos de tokens colados: se o grupo (com >1 token) recompõe uma fórmula
    # dopada/mista, ele sai inteiro, com a caixa e o hífen/parênteses originais,
    # como um único item do vocabulário. Senão, cada token do grupo segue o
    # tratamento normal (o mesmo de antes).
    with nlp.disable_pipes(*componentes_desnecessarios):
        doc = nlp(texto_doc)
    saida = []
    for grupo in agrupar_tokens_colados(doc):
        grupo = aparar_bordas(grupo)
        if not grupo:
            continue
        if len(grupo) > 1:
            texto_colado = ''.join(t.text for t in grupo)
            if parece_formula_colada(texto_colado):
                if somente_alpha and re.search(r'\d', texto_colado):
                    continue   # tok_alpha/tok_lema descartam números e códigos mistos, por definição
                saida.append(texto_colado)
                continue
        for t in grupo:
            if t.is_punct or t.is_space:
                continue
            if somente_alpha and not t.is_alpha:
                continue
            if remover_stop and t.is_stop:
                continue
            texto_original = aparar_pontuacao_textual(t.text)   # cobre 'eV.' fundido num token só
            if not texto_original:
                continue
            if preservar_caixa_texto(texto_original):
                saida.append(texto_original)   # símbolo/fórmula/unidade: nunca lematiza nem minusculiza
            else:
                if lematizar:
                    if t.tag_ in ('NNP', 'NNPS'):
                        # palavra em Título Maiúsculo (comum em subtítulos: 'Solar Cells.')
                        # engana o marcador gramatical do spaCy, que a lê como nome
                        # próprio e pula a lematização ('Cells' fica 'Cells', não 'cell').
                        # Relematiza forçando minúsculo pra pegar a forma certa.
                        bruto = nlp(texto_original.lower())[0].lemma_
                    else:
                        bruto = t.lemma_
                else:
                    bruto = texto_original
                saida.append(aparar_pontuacao_textual(bruto).lower())
    return saida

def tok_texto(doc):
    # Texto puro: descarta pontuação e espaços. Fórmulas dopadas
    # ('PbI3-xClx') saem inteiras, com a caixa e o hífen originais.
    return _tokenizar(doc, somente_alpha=False, lematizar=False, remover_stop=True)

def tok_alpha(doc):
    # Apenas tokens alfabéticos (descarta números e códigos mistos,
    # incluindo fórmulas com dígito — comportamento igual ao original).
    return _tokenizar(doc, somente_alpha=True, lematizar=False, remover_stop=False)

def tok_lema(doc):
    # Lema + alfabéticos + sem stop words. O mais agressivo.
    return _tokenizar(doc, somente_alpha=True, lematizar=True, remover_stop=True)

def tok_lema_formulas(doc):
    # Como tok_lema (normaliza plural/singular, tempo verbal etc.: 'cells' ->
    # 'cell'), mas sem o filtro 'só alfabético' — então fórmulas com dígito
    # ('MAPbI3-xClx', 'CsPbI2Br'...) saem inteiras, com a caixa original, em vez
    # de serem descartadas. Números soltos, sem nenhuma letra (sobras de citação,
    # '2', '3', '21.3'...), continuam fora: sozinhos não ajudam como feature.
    termos = _tokenizar(doc, somente_alpha=False, lematizar=True, remover_stop=True)
    return [t for t in termos if re.search(r'[A-Za-z]', t)]

# Teste rápido lado a lado, num trecho de um abstract real.
trecho = corpus[0][:200]
print('TRECHO ORIGINAL:'); print(trecho); print()
print('texto :', tok_texto(trecho)[:18]); print()
print('alpha :', tok_alpha(trecho)[:18]); print()
print('lema  :', tok_lema(trecho)[:18]); print()
print('lema+formulas:', tok_lema_formulas(trecho)[:18])

vectorizer_bow = CountVectorizer(
    tokenizer=tok_lema_formulas,
    token_pattern=None,
    lowercase=False,   # já lematizamos e colocamos em minúsculas dentro do tokenizer
)

bow = vectorizer_bow.fit_transform(corpus)

print(f'Dimensões da matriz BOW: {bow.shape}  (documentos × termos)')
print(f'Densidade: {100 * bow.nnz / (bow.shape[0] * bow.shape[1]):.2f}%')

# Os termos mais frequentes no corpus inteiro — devem ser os termos
# "de fundo" do tema (ex.: perovskite, solar, cell, efficiency...),
# que aparecem em quase todo abstract e por isso não discriminam nada
# entre os documentos. É exatamente o problema que o TF-IDF resolve
# no próximo bloco.
vocab_bow = vectorizer_bow.get_feature_names_out()
freq_total = np.asarray(bow.sum(axis=0)).ravel()
ordem = np.argsort(freq_total)[::-1]

print('\nOS 20 TERMOS MAIS FREQUENTES DO CORPUS')
for j in ordem[:20]:
    print(f'{vocab_bow[j]:<20} {freq_total[j]:>6}')


CHAVE = re.compile(r"[Yy]ield [Ss]trength")
VALOR = re.compile(r"\d+(?:\.\d+)?\s?[TGMKk]Pa\b")
LIGA = re.compile(r"\b[A-Z][a-z]?\d*(?:\.\d+)?"
                  r"(?:[A-Z][a-z]?\d*(?:\.\d+)?){4,8}\b")


def vale_a_chamada(texto: str) -> bool:
    # O resumo tem chance de conter o que procuramos?
    return bool(CHAVE.search(corpus) and VALOR.search(corpus) and LIGA.search(corpus))


candidatos = [a for a in corpus if vale_a_chamada(a)]

print(f"{len(abstracts)} resumos → {len(candidatos)} candidatos "
      f"({len(candidatos) / len(abstracts):.1%})")
print(f"chamadas economizadas: {len(abstracts) - len(candidatos)}")