# Mapeando a Eficiência de Células Solares de Perovskitas com Processamento de Linguagem Natura

Este repositório reúne os códigos desenvolvidos no projeto "Mapeando a Eficiência de Células Solares de Perovskita com Processamento de Linguagem Natural", voltado à extração e análise de informações sobre composição, processamento e eficiência de células solares de perovskita (PSCs) a partir da literatura científica.

### Objetivo

Esse projeto busca transformar informações dispersas em resumos científicos em um único conjunto de dados estruturado que relaciona:
- composição química da perovskita;
- eficiência de conversão energética (PCE)
- método de deposição;
- arquitetura do dispositivo.
Para isso, são combinadas técnicas de Processamento de Linguagem Natural (PLN), extração estruturada por LLM e aprendizado de máquina.

### Pipeline

O projeto é organizado nas seguintes etapas que se encontram nos códigos desse repositório:

1. Processamento dos resumos
   - limpeza e normalização dos textos;
   - tokenização com spaCy;
   - preservação de fórmulas químicas, unidades e siglas.

2. Reconhecimento e normalização de perovskitas
   - identificação de fórmulas do tipo ABX₃;
   - tratamento de composições mistas e dopadas;
   - uso de regex e ChemDataExtractor.

3. Extração estruturada
   - filtragem dos resumos candidatos;
   - extração de composição, eficiência, método de deposição e arquitetura.

4. Aprendizado de máquina
   - treinamento de um modelo Random Forest Regressor;
   - validação cruzada agrupada por artigo;
   - avaliação por MAE, RMSE e R²;
   - análise das previsões utilizando valores SHAP.

### Autoria

**L. H. Gavazza Pessôa**  
Ilum – Escola de Ciência  
Centro Nacional de Pesquisa em Energia e Materiais (CNPEM)

**L. Davoli**
Ilum – Escola de Ciência  
Centro Nacional de Pesquisa em Energia e Materiais (CNPEM)
