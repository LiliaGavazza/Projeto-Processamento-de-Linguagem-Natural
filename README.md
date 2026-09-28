# Mapeamento da Eficiência de Células Solares de Perovskitas com Processamento de Linguagem Natural

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

**[L. H. Gavazza Pessôa](https://github.com/LiliaGavazza)**  

Ilum – Escola de Ciência  
Centro Nacional de Pesquisa em Energia e Materiais (CNPEM)

**[L. Davoli](https://github.com/luiza160)**

Ilum – Escola de Ciência  
Centro Nacional de Pesquisa em Energia e Materiais (CNPEM)

### Orientação

**[J. Almeida](https://github.com/jamesmalmeida)**

### Referências

BABINCHAK, W. Michael; SUREWICZ, Witold K. Liquid–liquid phase separation and its mechanistic role in pathological protein aggregation. Journal of Molecular Biology, v. 432, n. 7, p. 1910–1925, 2020. DOI: 10.1016/j.jmb.2020.03.004.

BREIMAN, Leo. Random forests. Machine Learning, v. 45, n. 1, p. 5–32, 2001. DOI: 10.1023/A:1010933404324.

HAMEED, Talaat A.; ALBALAWI, Hind; MORSHEDY, Asmaa S.; LAHMAR, Abdelilah. Fundamentals, advances, and challenges of hybrid organic–inorganic perovskite solar cells. Discover Applied Sciences, v. 8, n. 1114, 2026. DOI: 10.1007/s42452-026-08916-3.

HONNIBAL, Matthew; MONTANI, Ines. spaCy 2: Natural language understanding with Bloom embeddings, convolutional neural networks and incremental parsing. 2017. Disponível em: https://spacy.io. Acesso em: 28 set. 2026.

JEON, Nam Joong et al. Solvent engineering for high-performance inorganic–organic hybrid perovskite solar cells. Nature Materials, v. 13, p. 897–903, 2014. DOI: 10.1038/nmat4014.

KIM, Jin Young et al. High-efficiency perovskite solar cells. Chemical Reviews, v. 120, n. 15, p. 7867–7918, 2020. DOI: 10.1021/acs.chemrev.0c00107.

LIU, Xiang et al. Perovskite-LLM: Knowledge-enhanced large language models for perovskite solar cell research. In: FINDINGS OF THE ASSOCIATION FOR COMPUTATIONAL LINGUISTICS: EMNLP 2025, 2025, Suzhou, China. Proceedings [...]. Suzhou: Association for Computational Linguistics, 2025. p. 494–518. DOI: 10.18653/v1/2025.findings-emnlp.27.

LUNDBERG, Scott M.; LEE, Su-In. A unified approach to interpreting model predictions. In: ADVANCES IN NEURAL INFORMATION PROCESSING SYSTEMS 30 (NEURIPS 2017), 2017. Proceedings [...]. Curran Associates, 2017. p. 4765–4774.

NLP DEMYSTIFIED. Natural Language Processing Demystified. 2021. GitHub repository. Disponível em: https://github.com/futuremojo/nlp-demystified. Acesso em: 28 set. 2026.

PEDREGOSA, Fabian et al. Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, v. 12, p. 2825–2830, 2011.

PRIEM, Jason; PIWOWAR, Heather; ORR, Richard. OpenAlex: A fully-open index of scholarly works, authors, venues, institutions, and concepts. arXiv preprint, arXiv:2205.01833, 2022.

SWAIN, Matthew C.; COLE, Jacqueline M. ChemDataExtractor: A toolkit for automated extraction of chemical information from the scientific literature. Journal of Chemical Information and Modeling, v. 56, n. 10, p. 1894–1904, 2016. DOI: 10.1021/acs.jcim.6b00207.

ZHANG, Lei et al. Fast exploring literature by language machine learning for perovskite solar cell materials design. Advanced Intelligent Systems, v. 6, n. 6, p. 2300678, 2024. DOI: 10.1002/aisy.202300678.
