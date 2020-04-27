//gcc mlpxor.c -lm -o exe

#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<string.h>
#include<time.h>

//BIBLIOTECA COM CLASSES E FUNCOES DO MLP
#include "mlp.h"

//FUNCTIONS
float funcActiv(float z); //ok
float derivFuncActiv(float z); //ok
void forward(int* inVector, model* arch); //ok
void backpropagation(int X[][inLength], int Y[][outputLength], model* arch);
void fillRandom(model* arch); //ok


//MAIN
int main(){
	int i, j, k=0;
	char c;
	FILE *dataset;
	model arch; //struct contendo os pesos, biases e resultados mais recentes dos neuronios do MLP

	
	//acessando arquivo contendo dataset
	dataset = fopen(inFile, "r");
	if(dataset == NULL){
		printf("error opening dataset\n");
		return -1;
	}

	//Contando quantidade de linhas do dataset = quantidade de "casos teste" para o MLP
	qtTrainCases = 0;
	while(fscanf(dataset, "%c", &c) != EOF){
		if(c == '\n'){
			qtTrainCases++;
		}
	};

	//matrizes de entradas de saidas para treinar o MLP
	int X[qtTrainCases][inLength], Y[qtTrainCases][outputLength];
	
	rewind(dataset);

	//Preenchendo matrizes com dados do dataset
	for(i=0;i<qtTrainCases;i++){
		for(j=0;j<inLength;j++){
			fscanf(dataset, "%d", &X[i][j]);
		}
		for(j=0;j<outputLength;j++){
			fscanf(dataset, "%d", &Y[i][j]);
		}
	}
	fclose(dataset);

	//definindo pesos e biases iniciais
	srand(time(0));
	fillRandom(&arch);

	backpropagation(X, Y, &arch);

	//TESTANDO MLP
	int vec[inLength] = {0};
	printf("Digite as %d entradas da rede:\n", inLength);
	for(i=0;i<inLength;i++){
		scanf("%d", &vec[i]);
	}

	forward(vec, &arch);
	printf("RESULTADO = %f\n", arch.outputResult[0]);


	return 0;
}