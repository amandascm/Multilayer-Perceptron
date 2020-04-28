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
int countLines(FILE* file);


//MAIN
int main(){
	int i, j, k=0, qtTestCases=0;
	char c;
	float vec[inLength] = {0};
	FILE *dataset, *test;
	model arch; //struct contendo os pesos, biases e resultados mais recentes dos neuronios do MLP

	
	//acessando arquivo contendo dataset
	dataset = fopen(trainFile, "r");
	if(dataset == NULL){
		printf("error opening dataset file\n");
		return -1;
	}

	//contando quantidade de linhas do dataset = quantidade de "casos treino" para o MLP
	qtTrainCases = countLines(dataset);

	//matrizes de entradas e de saidas para treinar o MLP
	float X[qtTrainCases][inLength], Y[qtTrainCases][outLength];

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

	//treinando MLP
	backpropagation(X, Y, &arch);

	//testando MLP

	test = fopen(testFile, "r");
	if(test == NULL){
		printf("error opening test file\n");
		return -1;
	}

	qtTestCases = countLines(test);

	for(i=0;i<qtTestCases;i++){

		for(j=0;j<inLength;j++){
			fscanf(test, "%f", &vec[j]);
		}

		forward(vec, &arch);
		
		printf("RESULTADO = [");
		for(k=0;k<outLength;k++){
			if(k == outLength-1){
				printf("%.1f]\n", arch.outResult[k]);
			}else{
				printf("%.1f, ", arch.outResult[k]);
			}
		}

	}
	fclose(test);


	return 0;
}