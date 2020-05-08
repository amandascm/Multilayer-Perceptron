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
	FILE *trainDataset, *testDataset;
	mlp mlp; //classe contendo os pesos, biases, resultados, informacoes e funcoes do MLP
	
	//acessando arquivo contendo trainDataset
	trainDataset = fopen(trainFile, "r");
	if(trainDataset == NULL){
		printf("error opening trainDataset file\n");
		return -1;
	}

	//contando quantidade de linhas do trainDataset = quantidade de "casos treino" para o MLP
	qtTrainCases = countLines(trainDataset);

	//matrizes de entradas e de saidas para treinar o MLP
	float X[qtTrainCases][inLength], Y[qtTrainCases][outLength];

	//preenchendo matrizes com dados do trainDataset
	for(i=0;i<qtTrainCases;i++){
		for(j=0;j<inLength;j++){
			fscanf(trainDataset, "%f", &X[i][j]);
		}
		for(j=0;j<outLength;j++){
			fscanf(trainDataset, "%f", &Y[i][j]);
		}
	}
	fclose(trainDataset);

	//treinando MLP
	mlp.backpropagation(X, Y, qtTrainCases);

	//testando MLP
	testDataset = fopen(testFile, "r");
	if(testDataset == NULL){
		cout << "error opening testDataset file\n";
		return -1;
	}

	qtTestCases = countLines(testDataset);

	for(i=0;i<qtTestCases;i++){
		for(j=0;j<inLength;j++){
			fscanf(testDataset, "%f", &vec[j]);
		}
		mlp.forward(vec);
		mlp.printResult();
	}
	fclose(testDataset);


	return 0;
}