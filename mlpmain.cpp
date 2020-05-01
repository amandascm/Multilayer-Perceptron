#include<iostream>
#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<string.h>
#include<time.h>

//BIBLIOTECA COM CLASSES E FUNCOES DO MLP
#include "mlp.hpp"

using namespace std;

int main(){
	int i, j, k=0, qtTestCases=0, qtTrainCases=0;
	char c;
	float vec[inLength] = {0};
	FILE *dataset, *test;
	model arch; //classe contendo os pesos, biases, resultados, informacoes e funcoes do MLP

	
	//acessando arquivo contendo dataset
	dataset = fopen(trainFile, "r");
	if(dataset == NULL){
		printf("error opening dataset file\n");
		return -1;
	}

	//contando quantidade de linhas do dataset = quantidade de "casos treino" para o MLP
	arch.setQtTrainCases(countLines(dataset));
	qtTrainCases = arch.getQtTrainCases();

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

	//treinando MLP
	arch.backpropagation(X, Y);

	//testando MLP
	test = fopen(testFile, "r");
	if(test == NULL){
		cout << "error opening test file\n";
		return -1;
	}

	qtTestCases = countLines(test);

	for(i=0;i<qtTestCases;i++){

		for(j=0;j<inLength;j++){
			fscanf(test, "%f", &vec[j]);
		}

		arch.forward(vec);
		arch.printResult();
	}
	fclose(test);


	return 0;
}