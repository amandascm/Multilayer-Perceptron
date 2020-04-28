# Multilayer Perceptron (MLP)


MLP com uma camada oculta e uma camada de saída. A quantidade de neurônios por camada e a quantidade de entradas da rede são definidas no arquivo *mlp.h* e podem ser alteradas ao redefinir os valores das seguintes constantes diretamente no arquivo:

  - *hidLength*
    Quantidade de neurônios da camada oculta
  - *outLength* 
    Quantidade de neurônios da camada de saída
  - *inLength*
    Quantidade de entradas da rede

Um arquivo *txt* contendo entradas e saídas esperadas é usado para treinar o MLP. O arquivo a ser usado para o treinamento da rede pode ser escolhido ao redefinir diretamente no arquivo *mlp.h* a string *trainFile* para "*train/nomeDoArquivo.txt*".

Um arquivo *txt* contendo entradas é usado para testar o MLP e obter resultados correspondentes às entradas. O arquivo a ser usado para o teste da rede pode ser escolhido ao redefinir diretamente no arquivo *mlp.h* a string *testFile* para "*test/nomeDoArquivo.txt*".
# Diretórios train e test
  - *train*
        Contém dataset's que podem ser usados para treinar a rede
  - *test* 
        Contém dataset's que podem ser usados para testar a rede

# Compilação e Execução
Para compilar, digite no terminal (do diretório onde está localizado o projeto):

    $ gcc mlp.c -lm -o exe 

Para executar, digite:

    $ ./exe
