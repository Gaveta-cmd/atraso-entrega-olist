# Dá para prever atraso de entrega na hora da compra?

Análise e modelo com a [base pública da Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce): onde o atraso se concentra, quanto ele pesa na nota e se dá para prevê-lo na hora da compra.

- **Notebook completo, com gráficos:** [notebooks/atraso_entrega.ipynb](notebooks/atraso_entrega.ipynb)
- **No Kaggle:** ainda não publicado

> **Sobre o uso de IA:** este projeto foi desenvolvido com apoio de um assistente de IA (Claude Code), que escreveu o código e rodou a análise, a partir do plano que defini.

## O resultado principal

**Nesta base, o que se sabe na hora da compra não basta para prever atraso em um período novo.** O modelo parecia bom na validação e caiu no teste:

| Avaliação (XGBoost) | ROC-AUC | PR-AUC | Atraso no período |
|---|---|---|---|
| Divisão aleatória (só para comparar) | 0,740 | 0,190 | 6,7% |
| Validação: abril e maio de 2018 | 0,743 | 0,156 | 5,5% |
| **Teste: junho a agosto de 2018** | **0,597** | **0,058** | 3,6% |

O intervalo de 95% do ROC-AUC no teste vai de 0,575 a 0,617. A regressão logística foi melhor no teste (0,669), mas o modelo foi escolhido antes, pela validação, e não troquei depois de ver o resultado. Ela também é fraca, e eu não colocaria nenhum dos dois em produção.

**Por que caiu:** o padrão mudou no período de teste. O prazo prometido mediano foi de 27,3 dias em junho para 13,5 em agosto, e o atraso do RJ foi de 14,7% no treino para 3,2% no teste. Em agosto, o ROC-AUC do modelo foi de 0,447, pior que um sorteio. Essa explicação veio depois de ver o teste, então é um diagnóstico, não uma prova.

**A lição:** a divisão aleatória e a validação temporal prometiam quase o mesmo número, e nenhuma antecipou a queda. Treinando mês a mês antes do teste, o ROC-AUC varia de 0,613 a 0,761, e os meses da validação estavam entre os melhores. Uma validação temporal única também pode enganar.

## O que a análise mostra

- **O atraso vem em picos:** novembro de 2017, fevereiro e março de 2018 têm 21,7% dos pedidos e 48,4% dos atrasos.
- **O RJ pesou muito, mas não durou:** tem 47,1% do excesso de atrasos da base toda. O efeito aparece mesmo na mesma faixa de distância, nas faixas onde o RJ tem mais pedidos.
- **Distância pesa principalmente nos picos:** fora deles, a taxa fica perto de 4% até 1.000 km; nos picos, vai de 4,6% a 26,5%.
- **Atraso e nota ruim andam juntos:** a nota média cai de 4,29 para 2,27, em todas as UFs (21 de 21) e categorias (29 de 29) com volume. Isso é um indício forte, não prova de causa.

## Decisões que mudam o resultado

- **Atraso por dia, não por data e hora.** A data prometida vem sem hora. Comparar direto marcaria como atraso quem recebeu no dia prometido (8,1% contra 6,8%).
- **Fim do período em 26/08/2018.** Na última semana de agosto, 10,8% dos pedidos ainda não tinham entrega, contra 2% a 3% nas anteriores.
- **Só features conhecidas na compra.** Datas de envio e entrega e a avaliação ficam de fora.
- **O treino só usa pedidos com a data prometida já vencida.** O critério usa o prazo, conhecido na compra; filtrar pela entrega tiraria justamente os atrasados.
- **Sem a feature `mês`,** ao contrário do plano original: ela piorou a validação (ROC-AUC de 0,665 contra 0,743).

## Limitações

- Só entram pedidos entregues. Somar os 1.097 enviados sem entrega levaria a taxa de 6,8% para 7,8%.
- O teste é um único período e não tem um pico do tamanho dos anteriores.
- A base não tem dados de transportadora, então a causa dos picos segue em aberto.
- É uma base de e-commerce, com prazos de semanas. As perguntas valem para delivery de comida; os números, não.

## Como rodar

Baixe a base do Kaggle e descompacte os CSVs em uma pasta. No Kaggle, o notebook encontra os dados sozinho.

```bash
pip install -r requirements.txt
OLIST_DIR=/caminho/dos/csvs python notebooks/gerar_notebook.py
```

O arquivo editável é `notebooks/atraso_entrega.py`; o `gerar_notebook.py` monta e executa o `.ipynb` a partir dele. As validações interrompem a execução se os dados não baterem.
