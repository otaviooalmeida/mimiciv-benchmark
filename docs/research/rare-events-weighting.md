# Ponderação de eventos raros no MIMIC-IV / TDSTF

## Resumo e escopo

**Recomendação:** começar com ponderação moderada de quedas rápidas observadas, separada da raridade do valor absoluto, e avaliar eventos nas trajetórias probabilísticas completas. Comparar loss ponderada e oversampling de janelas em experimentos separados. Não começar com SMOTE ou substituir diretamente a loss de ruído por uma loss de regressão de sinais.

Esta é uma pesquisa com leitura estática do repositório, não um diagnóstico causal do checkpoint. Não há dados processados nem checkpoints neste checkout; não foram reproduzidas as previsões relatadas nem medidos ganhos. Os valores de hiperparâmetros abaixo são propostas de ablação, não resultados publicados para este projeto. Os trabalhos citados fundamentam técnicas gerais; não demonstram que elas resolvam este caso específico no MIMIC-IV.

## 1. O que o código atual faz

Fontes locais:

- [`preprocess/windowing.py`](../../preprocess/windowing.py): 60 minutos de histórico, previsão dos 20 seguintes e stride de 20 minutos. Não interpola observações. Exige duas observações futuras de **algum** alvo, não de todos os sinais.
- [`dataset.py`](../../dataset.py): representa observações como `(variável, minuto, valor, máscara)`, com diferentes sinais intercalados; limita o histórico a `diffusion.size` observações. O DataLoader de treino usa shuffle uniforme.
- [`diff.py`](../../diff.py), `TDSTF.forward`: amostra um passo de difusão e ruído gaussiano; aprende a prever esse ruído. A loss é `sum(mask * (noise - predicted)**2) / sum(y_len)`, sem ponderação por raridade.
- [`exe.py`](../../exe.py), `calc_metrics`: calcula MSE usando a **mediana** das amostras da difusão. A seleção do checkpoint usa uma aproximação agregada de CRPS, com apenas 5 amostras e quantis 0,25/0,50/0,75 na validação.
- [`infer.py`](../../infer.py): exporta média, mediana, intervalos marginais e `posterior_samples.npz`. O gráfico de distribuição compara observações reais com **medianas previstas**, não com a distribuição de todas as amostras geradas.

Consequências verificáveis da definição do objetivo:

1. Cada observação válida tem o mesmo peso. Janelas/sinais com mais observações contribuem mais para a loss; isso não equivale a dar o mesmo peso por janela, sinal ou paciente.
2. Não se deve confundir a MSE de **ruído** do treinamento com uma MSE aplicada diretamente aos sinais vitais. O fato de uma MSE pontual favorecer uma média condicional não demonstra colapso de uma distribuição de difusão.
3. Uma mediana pontual pode não mostrar um evento de baixa probabilidade. Em um exemplo hipotético, com 80% de trajetórias estáveis e 20% com queda, a mediana pode continuar estável. Mesmo com muitas trajetórias contendo queda, instantes diferentes podem fazer a agregação ocultá-la. Isso é uma propriedade da agregação, não evidência de que está acontecendo aqui.
4. A distribuição marginal das medianas condicionais não precisa reproduzir a distribuição dos valores observados. Forçar apenas essa semelhança pode piorar previsões. A escolha entre média, mediana, quantis e distribuição completa precisa acompanhar a tarefa e a métrica [7].

## 2. Definir o que é raro

Há pelo menos três objetivos diferentes:

| Objetivo | Exemplo ilustrativo | Descritor |
|---|---|---|
| Valor extremo | Pressão persistentemente baixa | Valor futuro por sinal |
| Transição rápida | Pressão cai rapidamente, mesmo terminando em faixa comum | Variação e tempo entre observações |
| Evento relevante | Queda sustentada ou cruzamento de um limiar escolhido para o estudo | Regra temporal/clínica explícita |

Uma loss que dá mais peso a valores baixos pode aprender melhor hipotensão persistente sem aprender a antecipar o **início** de uma queda. Para o relato apresentado, priorizar a segunda linha.

### Construção de um evento de queda

Para cada janela e **cada sinal separadamente**, ordenar as observações válidas por tempo. Para uma observação futura `j`, usar a observação anterior do mesmo sinal; para a primeira observação futura, incluir a última observação do histórico quando disponível e suficientemente recente.

```text
amplitude_j = max(0, y_anterior - y_j)
velocidade_j = amplitude_j / (tempo_j - tempo_anterior)
queda_j = (velocidade_j >= q_v) AND (amplitude_j >= delta_v)
```

- Calcular somente para `0 < intervalo <= gap_max_v`. Uma diferença entre medidas separadas por muito tempo não localiza uma queda súbita.
- Estimar `q_v`, por exemplo, pelo percentil 95 das velocidades positivas de queda **do treino**, por sinal. Escolher `delta_v` e `gap_max_v` segundo resolução observacional e objetivo do estudo; não assumir que o mesmo limite serve para temperatura, HR e pressão.
- Quantis estatísticos definem raridade, não relevância clínica. Limiares clínicos requerem justificativa específica; nenhum número deste documento é um protocolo assistencial.
- Sem par elegível, a dinâmica é desconhecida. Manter o ponto na loss-base, mas não tratá-lo automaticamente como um negativo confiável em métricas de evento.
- Se houver resolução suficiente, experimentar confirmação em outra observação ou duração mínima, sem excluir sistematicamente eventos verdadeiros curtos.
- Usar valores em unidades originais, ou escalas por variável ajustadas no treino. Não misturar bpm, mmHg e °C numa densidade única.

**Importante:** não aplicar `torch.diff` diretamente na dimensão dos triplets. Posições vizinhas podem pertencer a sinais diferentes. Preservar metadados dos pares e devolver os pesos para as posições originais.

Usar o futuro verdadeiro para definir pesos/rótulos **durante o treino** é supervisão normal. Não usar essa informação para selecionar entradas, condicionar o modelo em produção ou escolher trajetórias na inferência.

## 3. Estratégias de implementação, em ordem sugerida

### A. Loss de difusão ponderada pelo evento — primeira opção

Manter o objetivo de previsão de ruído, alterando somente a importância dos termos observados:

\[
L = \frac{\sum_{i,j} m_{ij} w_{ij}(\epsilon_{ij}-\hat\epsilon_{ij})^2}
         {\sum_{i,j} m_{ij} w_{ij}}.
\]

`i` representa a janela e `j` uma observação-alvo, não necessariamente um minuto de uma grade regular.

Primeiro experimento simples:

```text
peso = 1 + 2 * indicador_de_queda
```

Assim, observações marcadas recebem peso 3, as demais peso 1. Comparar pesos de evento 2, 3 e 5; não são ótimos conhecidos. Uma segunda ablação pode adicionar peso para caudas:

```text
peso = clip(1 + 2 * indicador_de_queda + indicador_de_cauda, 1, 5)
```

É uma proposta de engenharia inspirada em aprendizado sensível a custo, não uma reprodução literal de DenseLoss [1]. Não zerar o peso dos períodos estáveis: eles ensinam a evitar falsos alarmes. Uma rampa gradual do peso ao longo do treino é uma opção posterior, não necessária para a primeira ablação.

#### Encaixe no código

1. Construir estatísticas de raridade exclusivamente com o conjunto de treino, após fixar a divisão por paciente.
2. Em `dataset.py`, pré-calcular `loss_weight` alinhado com os alvos. Para detectar o primeiro par histórico/futuro, usar o histórico completo antes da seleção que remove observações; auditar separadamente se o contexto correspondente é preservado para o modelo.
3. Em `diff.py`, aplicar pesos à loss de ruído. Se forem calculados no `forward`, copiar o alvo limpo **antes** de `samples_y[:, 2]` ser substituído pelo alvo ruidoso. Nunca estimar raridade usando o ruído artificial.
4. Salvar configuração, limiares e estatísticas junto ao experimento.

Esboço, não código já implementado:

```python
# residual e mask_y são os tensores já existentes em TDSTF.forward.
# loss_weight: [batch, observações_alvo], pré-calculado com os alvos limpos.
w = batch["loss_weight"].to(residual.device, dtype=residual.dtype).detach()
weighted_mask = w * mask_y
loss = (residual.square() * weighted_mask).sum() / weighted_mask.sum().clamp_min(1e-8)
```

Com pesos 1, essa redução deve coincidir com a loss atual, assumindo `sum(mask_y) == sum(info[:, 2])`.

**Alternativa por janela:** definir `E_i = any(queda_j)` e dar peso à janela inteira. Preserva a importância conjunta do cenário, mas também aumenta o peso dos outros sinais e trechos estáveis dessa janela. Se quiser peso igual por janela, calcular primeiro a média mascarada de cada janela e depois a média ponderada entre janelas. Isso muda também a política de agregação; testar a média por janela sem ponderação como controle separado.

### B. DenseWeight / LDS — raridade contínua

**DenseWeight/DenseLoss** estima a densidade dos alvos por KDE e atribui mais importância a regiões menos frequentes. O artigo usa densidade reescalada `p'(z)` e, em essência [1]:

\[
u(z)=\max(1-\alpha p'(z),\varepsilon),\quad
w(z)=u(z)/\operatorname{mean}_{train}(u).
\]

Há uma implementação dos autores [2]. `alpha=0` desativa a ponderação. Não confundir essa fórmula com o inverso cru da densidade.

**Label Distribution Smoothing (LDS)** suaviza o histograma dos alvos antes de estimar a importância, aproveitando a proximidade entre valores de regressão [3]. Uma variante prática de reweighting é:

\[
w(z) \propto (\tilde n_{bin(z)}+\varepsilon)^{-\gamma}.
\]

Experimentar `gamma=0.5` antes de `1`, limitar pesos e normalizar sua escala. O README dos autores exemplifica inverso da frequência suavizada; o expoente moderado e clipping aqui são escolhas propostas para a ablação.

Aplicações neste projeto:

- `z = valor futuro`, para melhorar caudas dos valores absolutos;
- `z = variação/velocidade`, para melhorar transições raras;
- descritores separados por direção, para distinguir quedas de subidas;
- estatísticas por sinal; não começar com KDE multivariada em toda a trajetória esparsa.

É possível ponderar somente quedas com uma função de relevância explícita. Isso é diferente de ponderar raridade simetricamente. Ajustar densidades, bins, suavização, clipping e escalas no treino; usar validação para selecionar hiperparâmetros.

### C. Oversampling de janelas reais com eventos

Modificar somente o sampler de treino em `dataset.py`. O `WeightedRandomSampler` recebe um peso por elemento do dataset e suporta amostragem com reposição [4].

```python
sampler = WeightedRandomSampler(
    weights=window_weights,
    num_samples=len(train_data),
    replacement=True,
)
train_loader = DataLoader(train_data, batch_size=batch_size, sampler=sampler)
# Não combinar sampler com shuffle=True.
```

Um peso 3 aumenta a probabilidade relativa de selecionar a janela; não garante uma proporção fixa de eventos por batch. Se a necessidade for essa garantia, usar um batch sampler estratificado, com fração escolhida na validação. Não impor 50/50 automaticamente.

- Reamostrar janelas completas, não pontos independentes.
- Manter validação e teste na distribuição natural.
- Monitorar diversidade de pacientes/internações; duplicar muitas janelas do mesmo paciente não acrescenta diversidade clínica.
- Históricos se sobrepõem e janelas da mesma internação são correlacionadas. Bootstrap e intervalos de confiança devem agrupar por paciente.
- Comparar **oversampling com loss normal** contra **amostragem uniforme com loss ponderada**. Aplicar o mesmo fator nas duas partes multiplica a ênfase no risco esperado e dificulta atribuir ganhos.

**Cautela probabilística:** reamostragem ou pesos dependentes do desfecho mudam o objetivo. Para um peso escalar de janela `w(x,y)`, a distribuição reponderada é proporcional a `w(x,y) p(y|x)`, após normalização. Portanto, maior sensibilidade pode vir com superestimação da frequência de quedas. Pesos por coordenada modificam o objetivo de forma ainda menos simples. Avaliar e, se necessário, recalibrar em validação com prevalência natural. Isso é uma dedução do objetivo, não um ganho comprovado neste modelo.

### D. Loss auxiliar de dinâmica/forma — etapa posterior

Uma proposta simples para um modelo de regressão direta é penalizar erro na velocidade:

\[
L_{dyn} = \operatorname{mean}_{pares\ válidos}
\rho\left(
\frac{\hat y_j-\hat y_k}{t_j-t_k}
-
\frac{y_j-y_k}{t_j-t_k}
\right),
\]

em que `rho` pode ser Huber e os pares pertencem ao mesmo sinal. A regularização deve comparar **dinâmica prevista com real**, não penalizar apenas a magnitude da derivada prevista — isso incentivaria ainda mais suavização.

No TDSTF, `predicted` estima ruído, não `y`. Uma adaptação experimental exigiria primeiro reconstruir o alvo limpo:

\[
\hat y_0=(y_t-\sqrt{1-\bar\alpha_t}\,\hat\epsilon)/\sqrt{\bar\alpha_t}.
\]

Essa estimativa pode ser instável em passos com muito ruído; controlar por SNR/passo, usar coeficiente auxiliar pequeno e verificar impacto na diversidade/calibração. Não substituir cegamente a loss de difusão por loss de sinais.

**DILATE** foi proposta especificamente para forma e localização temporal de mudanças em previsão multihorizonte [5]. É uma referência pertinente se a queda existe mas sua forma/instante está errado. Contudo, não é uma substituição direta para o denoising atual: seria necessário adaptar reconstrução, sinais separados, irregularidade e máscaras. Alinhamento temporal excessivamente permissivo também pode esconder atrasos importantes.

### E. Prever probabilidade de evento, não forçar a mediana a cair

O modelo já gera trajetórias. Antes de adicionar uma cabeça classificadora, estimar:

\[
\hat P(E\mid x)=\frac{1}{S}\sum_{s=1}^{S}\mathbf{1}\{E(\hat y^{(s)},x)\}.
\]

Aplicar a regra de evento a cada trajetória **completa**, mantendo o mesmo índice `s` entre instantes. Não montar uma trajetória sorteando quantis ou amostras independentemente em cada ponto. Não escolher a trajetória mais próxima da verdade após observar o futuro.

Isso permite distinguir:

- mediana estável, mas risco de queda relevante;
- distribuição inteira concentrada em persistência;
- trajetórias instáveis sem capacidade de discriminar quais pacientes terão eventos.

O arquivo `posterior_samples.npz` já preserva as amostras necessárias. Começar com 100 trajetórias por janela, como a inferência atual, e verificar sensibilidade ao número de amostras; eventos de probabilidade muito pequena exigem mais amostras.

No benchmark atual, as posições consultadas e a máscara vêm das observações futuras do dataset. Portanto, essa estimativa descreve eventos nessa grade observacional, não qualquer queda que possa ter ocorrido entre medidas. Para uso prospectivo, definir a grade de consulta sem conhecer os registros futuros e validar a cobertura correspondente; ausência de observação não comprova ausência de evento.

Se necessário, uma cabeça auxiliar de evento condicionada **apenas ao histórico** pode ser testada depois. Losses classificatórias ponderadas também exigem cuidado com calibração; uma cabeça que veja o alvo futuro ruidoso durante treino não representa automaticamente um detector prospectivo.

## 4. Auditorias importantes antes de atribuir o problema à raridade

São verificações propostas, não causas confirmadas:

1. **Recência do histórico selecionado.** `generate_windows` ordena observações por minuto. Na seleção de `triplet_generate`, a busca por cada sinal-alvo percorre o início do histórico e remove o primeiro item encontrado. Com excesso de observações, isso privilegia observações antigas desses sinais, em vez de garantir as últimas antes da previsão. Medir a idade da última observação por sinal antes/depois da seleção. Comparar uma política que sempre preserve as últimas observações e mantenha parte do contexto antigo, com o mesmo orçamento de 60 pontos.
2. **MIMIC-IV clínico não equivale a waveform.** `chartevents` contém registros do prontuário, e `charttime` é o melhor proxy do instante da medida, enquanto `storetime` é o instante de entrada/validação [6]. A base MIMIC-IV Waveform é outro recurso, com sinais de monitor e numerics [8]. Agrupar em bins de minuto não cria observações a cada minuto. Medir gaps reais por sinal e a fração de eventos que é possível rotular.
3. **Agregação e fonte do sinal.** `step_2.py` usa média por `(internação, minuto, variável)`. `step_1.py` combina diferentes itemids sob a mesma variável, inclusive modalidades distintas de pressão e saturação. Auditar se aparentes saltos são trocas de fonte/medida e se a média apaga extremos. Não dar peso máximo automaticamente a qualquer outlier.
4. **Normalização.** `step_4.py` ajusta média/desvio usando treino **e validação**. Para uma avaliação estritamente independente, ajustar no treino apenas; também ajustar nele as estatísticas de raridade. Não há evidência de que esse detalhe cause o sintoma relatado.
5. **Covariáveis disponíveis no instante da previsão.** Auditar se intervenções relevantes e últimas medidas chegam de fato ao histórico após a seleção. Não usar intervenções futuras realizadas como entradas conhecidas. Uma queda causada por informação indisponível não se torna determinística só por aumentar seu peso.
6. **Separação e seleção de modelo.** Preservar split por paciente já usado, fixar sua semente/manifesto e comparar experimentos no mesmo split. O CRPS global sozinho pode selecionar um checkpoint que perca em eventos raros. Manter também uma métrica global e uma restrição de falsos alarmes, em vez de escolher somente pelo desempenho nas quedas.

## 5. Plano experimental mínimo

### Antes de retreinar

- Gerar estatísticas de cobertura, gaps, distribuição de quedas e número de pacientes com eventos por sinal.
- Separar extremos estáveis de transições rápidas.
- Inspecionar trajetórias completas e probabilidades de evento em exemplos positivos **e negativos**, escolhidos sem cherry-picking.
- Comparar persistência usando a última medida realmente disponível ao modelo, e também persistência com o histórico completo como auditoria da seleção. Reportar disponibilidade por sinal; quando possível, incluir tendência linear simples como segundo baseline.

### Ablações

| Experimento | Mudança isolada |
|---|---|
| A | Baseline atual |
| B | Preservação de observações recentes, sem pesos |
| C | Baseline + peso 3 para quedas |
| D | Baseline + LDS/DenseWeight em velocidades |
| E | Baseline + oversampling de janelas de queda, loss original |
| F | Melhor opção anterior + peso moderado para valores de cauda |

Só combinar mudanças após entender os efeitos individuais. Usar várias seeds, mesmo orçamento de atualizações e o mesmo split. Se a agregação passar a ser por janela/sinal, incluir um controle com essa agregação e pesos uniformes.

### Métricas necessárias

- MAE/RMSE por sinal, em unidades originais; média como estimador pontual para MSE e mediana para MAE, mantendo a métrica histórica identificada para comparabilidade [7].
- Erro em períodos estáveis, em extremos estáveis e em transições rápidas, com tamanhos dos subgrupos.
- Skill contra persistência no total e nos eventos, usando os mesmos pontos observados.
- Amplitude da queda e erro do instante de início, somente na resolução sustentada pelos dados. Medir timing nos eventos detectados e reportar também os eventos perdidos.
- AUPRC, recall a precisão/FPR fixada, Brier score e curva de calibração para probabilidade de evento, com prevalência natural.
- Falsos alarmes por hora/paciente apenas se houver uma definição longitudinal válida de alarme, exposição observada e agrupamento de alertas repetidos. Como fallback, usar FPR por janela elegível.
- CRPS, cobertura e largura dos intervalos na distribuição natural. Resultados condicionados a eventos são diagnósticos úteis, mas não equivalem a calibração marginal.
- Intervalos de confiança agrupados por paciente, não por ponto ou janela independente.

Não considerar sucesso apenas maior variância visual ou melhor recall. O objetivo é melhorar a discriminação e/ou o erro em eventos sem degradação inaceitável de falsos alarmes, calibração e períodos normais.

### Testes para uma implementação futura

- Peso 1 reproduz a loss-base.
- Padding não contribui; pesos e loss permanecem finitos.
- Deltas nunca cruzam sinais, internações ou gaps proibidos.
- O primeiro alvo usa corretamente a última medida elegível do histórico.
- Pesos não dependem do ruído nem do passo sorteado da difusão.
- Estatísticas de pesos nunca são ajustadas com validação/teste.
- Preservar split por paciente e registrar parâmetros do sampler.

## 6. O que não priorizar

- **SMOTE/SMOGN diretamente em sequências achatadas:** SMOGN existe para regressão desbalanceada [9], mas preservar temporalidade, covariáveis e coerência fisiológica exigiria uma adaptação. Reamostrar janelas reais é uma primeira experiência mais controlável.
- **Balanced MSE diretamente em `noise - predicted`:** o trabalho trata desbalanceamento de alvos de regressão [10]. Aqui, a raridade relevante é do sinal limpo/evento, não do ruído gaussiano sorteado. Não é um substituto automático.
- **Focalização apenas em erro alto:** casos difíceis podem ser artefatos ou ruído irreduzível; dificuldade não equivale a raridade relevante.
- **Forçar curvas a oscilar:** aumentar variabilidade não prova antecipação nem generalização.
- **Balancear o teste:** ocultaria o desempenho sob a prevalência de interesse.

## Fontes primárias

1. Steininger et al. (2021). **Density-based weighting for imbalanced regression**. Artigo original, especialmente seção 3 e equações 2–5. https://link.springer.com/article/10.1007/s10994-021-06023-5
2. Steininger. **DenseWeight**, implementação dos autores e README. https://github.com/SteiMi/denseweight
3. Yang et al. (ICML 2021). **Delving into Deep Imbalanced Regression**. https://proceedings.mlr.press/v139/yang21m.html — implementação e exemplos de LDS/FDS: https://github.com/YyzHarry/imbalanced-regression
4. PyTorch. **WeightedRandomSampler**, documentação oficial. https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.WeightedRandomSampler
5. Le Guen & Thome (NeurIPS 2019). **Shape and Time Distortion Loss for Training Deep Time Series Forecasting Models**. https://papers.nips.cc/paper/8672-shape-and-time-distortion-loss-for-training-deep-time-series-forecasting-models
6. MIT-LCP. **MIMIC-IV: chartevents**, documentação oficial. https://mimic.mit.edu/docs/IV/modules/icu/chartevents.html
7. Gneiting (JASA 2011). **Making and Evaluating Point Forecasts**. https://doi.org/10.1198/jasa.2011.r10138 — manuscrito do autor: https://arxiv.org/abs/0912.0902
8. Moody et al. **MIMIC-IV Waveform Database v0.1.0**, documentação e descrição dos dados. Esta versão consultada é um preview de 200 registros/198 pacientes, não um substituto de mesma cobertura para o MIMIC-IV clínico. https://physionet.org/content/mimic4wdb/0.1.0/
9. Branco et al. (PMLR 2017). **SMOGN: a Pre-processing Approach for Imbalanced Regression**. https://proceedings.mlr.press/v74/branco17a.html
10. Ren et al. (CVPR 2022). **Balanced MSE for Imbalanced Visual Regression**. https://openaccess.thecvf.com/content/CVPR2022/html/Ren_Balanced_MSE_for_Imbalanced_Visual_Regression_CVPR_2022_paper.html
