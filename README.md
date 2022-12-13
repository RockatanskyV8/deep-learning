# deep-learning

 Treinamento de Rede Neural para previsões com os dados de `points.txt`

### Arquivos:
* `Reader.py`: Leitura e formatação dos dados.
* `Modelos.py`: Modelos disponíveis para uso(Neste caso foi feito uma estrutura para criação de camadas dinamicamente).
* `Plots.py`: Funções de plotagem para exibir as informações e plotar graficos.
* `Treinamento.py`: Classe para execução da função de treinamento


* `teste.py` : Teste de classificação
* `teste2.py` : Teste de Regressão
* `teste3.py` : Teste de Validação cruzada

# Classificação

[Nesse notebook](https://colab.research.google.com/drive/1JvqY4rVoe326ZYBHM90SHk3Km3kVIu-H#scrollTo=g5u7105CUSYf), fiz os teste usando diferentes números de neurônios e de camadas. 
Sempre aumentando número de camadas e quantidade de neurônios até chegar nesse modelo:

    GeradorRede(
      (layers): ModuleList(
        (0): Linear(in_features=3, out_features=1500, bias=True)
        (1): ReLU()
        (2): Dropout(p=0.5, inplace=False)
        (3): Linear(in_features=1500, out_features=1000, bias=True)
        (4): ReLU()
        (5): Dropout(p=0.5, inplace=False)
        (6): Linear(in_features=1000, out_features=500, bias=True)
        (7): ReLU()
        (8): Dropout(p=0.5, inplace=False)
        (9): Linear(in_features=500, out_features=250, bias=True)
        (10): ReLU()
        (11): Dropout(p=0.5, inplace=False)
        (12): Linear(in_features=250, out_features=100, bias=True)
        (13): ReLU()
        (14): Dropout(p=0.5, inplace=False)
        (15): Linear(in_features=100, out_features=10, bias=True)
        (16): ReLU()
        (17): Linear(in_features=10, out_features=2, bias=True)
        (18): Softmax(dim=-1)
      )
    )

Com esse modelo atingi uma acurácia de `0.7746666666666666`


## Regressão

Com os a regressão tive dificuldades de implementar no colab, por algum motivo eu sempre recebi algum erro de conexão no meio da execução, então eu testei localmente.
Eu comecei testando o mesmo modelo que eu cheguei ao testar a Classificação.

* Learning rates: [0.0001, 0.001, 0.01, 0.1]
* Batch_sizes:    [65, 90, 120, 135]
* Epocas:         2000
* 5 tentativas

Dessa forma atingi a acurácia `0.7713333333333333`, um pouco menor do que com a implementação linear e tomou um dia inteiro.

Então decidi testar de outra forma no de implementar o modelo pelo colab, que foi carregando o `best_global_model` dentro do loop de regressão.


        for initializations in range(0, self.retries):
            for learning_rate in learning_rates:
                for batch_size in self.batch_sizes:

                    print(f'----------\nAtempt: {initializations}\nLearning rate: {learning_rate}\nBatch size: {batch_size}\n')

                    model = GeradorRede(input_features, layers)

                    ########################################### Dessa forma ###################################################
                    if (best_global_model is not None):
                        model.load_state_dict(torch.load(best_global_model)) 

Por padrão, a variável do modelo é `None`, mas, caso atribuida durante a implementação ele carregaria o modelo existente.
Os testes foram feitos 
[nesse notebook](https://colab.research.google.com/drive/128a011xENsVt4ezK8BCBzJSrx9u66hMl#scrollTo=d-s6Gcjltevs).

O modelo que usei para esse teste foi:

    Total training time: 0:07:10.391927
    GeradorRede(
      (layers): ModuleList(
        (0): Linear(in_features=3, out_features=500, bias=True)
        (1): ReLU()
        (2): Dropout(p=0.5, inplace=False)
        (3): Linear(in_features=500, out_features=250, bias=True)
        (4): ReLU()
        (5): Dropout(p=0.5, inplace=False)
        (6): Linear(in_features=250, out_features=100, bias=True)
        (7): ReLU()
        (8): Dropout(p=0.5, inplace=False)
        (9): Linear(in_features=100, out_features=10, bias=True)
        (10): ReLU()
        (11): Linear(in_features=10, out_features=2, bias=True)
        (12): Softmax(dim=-1)
      )
    )

O teste "ótimo" estava demorando muito para rodar.

Dessa forma a cada vez que eu fiz uma nova tentativa, alterei os valores de learning_rates e batch_sizes. 
Eu usei listas de comprimentos menores, assim demoraria menos para que a tentativa fosse concluida.

Dessa forma atingi a acurácia `0.778`(mesmo com menos camadas e neurônios) e uma loss `0.5349240899085999`. 
