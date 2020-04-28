#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<string.h>
#include<time.h>

//BIBLIOTECA COM CLASSES E FUNCOES DO MLP
#include "mlp.h"

//FUNCTIONS
float activFunc(float z);
float activFuncDeriv(float z);
void forward(float* inVector, model* arch);
void backpropagation(float X[][inLength], float Y[][outLength], model* arch);
void fillRandom(model* arch);


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

	//contando quantidade de linhas do dataset = quantidade de "casos treino" para o MLP
	qtTrainCases = 0;
	while(fscanf(dataset, "%c", &c) != EOF){
		if(c == '\n'){
			qtTrainCases++;
		}
	};

	//matrizes de entradas e de saidas para treinar o MLP
	float X[qtTrainCases][inLength], Y[qtTrainCases][outLength];
	
	rewind(dataset);

	//preenchendo matrizes com dados do dataset
	for(i=0;i<qtTrainCases;i++){
		for(j=0;j<inLength;j++){
			fscanf(dataset, "%f", &X[i][j]);
		}
		for(j=0;j<outLength;j++){
			fscanf(dataset, "%f", &Y[i][j]);
		}
	}
	fclose(dataset);

	//definindo pesos e biases iniciais
	srand(time(0));
	fillRandom(&arch);

	backpropagation(X, Y, &arch);

	//testando MLP
	float vec[inLength] = {0};
	printf("Digite as %d entradas da rede:\n", inLength);
	for(i=0;i<inLength;i++){
		scanf("%f", &vec[i]);
	}

	forward(vec, &arch);
	
	printf("RESULTADO = [");
	for(i=0;i<outLength;i++){
		if(i == outLength-1){
			printf("%f]\n", arch.outResult[i]);
		}else{
			printf("%f, ", arch.outResult[i]);
		}
	}


	return 0;
}