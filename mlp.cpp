#include<iostream>
#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<string.h>
#include<time.h>

//BIBLIOTECA COM CLASSES DO MLP
#include "mlp.hpp"

using namespace std;

//FUNCTION BODY
int countLines(FILE* file){
	int qtLines = 0;
	char c;
	while(fscanf(file, "%c", &c) != EOF){
		if(c == '\n'){
			qtLines++;
		}
	}
	rewind(file);
	return qtLines;
}

//MLP CLASS
mlp::mlp(){
	//preenche matrizes de pesos e biases com numeros pseudoaleatorios entre -0.5 e 0.5
	int i, j;
	srand(time(0));
	for(i=0;i<hidLength;i++){
		for(j=0;j<(inLength+1);j++){
			matH[i][j] = 2.0f * ((float)rand() / (2.0f * (float)RAND_MAX)) - 0.5f;
		}
	}
	for(i=0;i<outLength;i++){
		for(j=0;j<(hidLength+1);j++){
			matO[i][j] = 2.0f * ((float)rand() / (2.0f * (float)RAND_MAX)) - 0.5f;
		}
	}
}

void mlp::setQtTrainCases(int x){
	qtTrainCases = x;
}

int mlp::getQtTrainCases(){
	return qtTrainCases;
}

void mlp::printResult(){
	int k;
	printf("RESULTADO = [");
	for(k=0;k<outLength;k++){
		if(k == outLength-1){
			printf("%.1f]\n", outResult[k]);
		}else{
			printf("%.1f, ", outResult[k]);
		}
	}
}

float mlp::activFunc(float z){
	//sigmoide
	return (1.0/(1.0 + expf(-z)));
}

float mlp::activFuncDeriv(float z){
	//derivada sigmoide
	return (z*(1.0 - z));
}

void mlp::forward(float* inVector){
	int i, j;
	float totalH = 0, totalO = 0;

	for(i=0;i<hidLength;i++){
		totalH = 0;
		for(j=0;j<(inLength);j++){
			totalH += matH[i][j] * inVector[j]; // + w*x
		}
		 totalH += matH[i][inLength]; // + bias
		 hidResult[i] = activFunc(totalH);
	}

	for(i=0;i<outLength;i++){
		totalO = 0;
		for(j=0;j<hidLength;j++){
			totalO += matO[i][j] * hidResult[j]; // + w*z
		}
		totalO += matO[i][hidLength]; // + bias
		outResult[i] = activFunc(totalO);
	}
}

void mlp::backpropagation(float X[][inLength], float Y[][outLength]){
	int i, j, k;
	float inVector[inLength] = {0};
	float erro, sum, erroMLP = 2*threshold;
	float deltaHid[hidLength], deltaOut[outLength];
	
	while(erroMLP > threshold){

		erroMLP = 0;

		for(i=0;i<qtTrainCases;i++){

			//feedforward para a linha i do dataset
			for(j=0;j<inLength;j++){
				inVector[j] = X[i][j];
			}
			forward(inVector);

			//calculo do delta de cada neuronio da camada output
			for(j=0;j<outLength;j++){
				erro = Y[i][j] - outResult[j];
				erroMLP += pow(erro, 2);
				deltaOut[j] = -2 * erro * activFuncDeriv(outResult[j]);
			}

			//calculo do delta de cada neuronio da camada hidden
			for(j=0;j<hidLength;j++){
				sum = 0;
				for(k=0;k<outLength;k++){
					sum += deltaOut[k] * matO[k][j];
				}
				deltaHid[j] = sum * activFuncDeriv(hidResult[j]);
			}

			//atualizacao de pesos e bias da camada hidden
			for(j=0;j<hidLength;j++){
				for(k=0;k<inLength;k++){
					matH[j][k] = matH[j][k] - learningRate * deltaHid[j] * inVector[k];
				}
				matH[j][k] = matH[j][k] - learningRate * deltaHid[j];
			}

			//atualizacao de pesos e bias da camada output
			for(j=0;j<outLength;j++){
				for(k=0;k<hidLength;k++){
					matO[j][k] = matO[j][k] - learningRate * deltaOut[j] * hidResult[k];
				}
				matO[j][hidLength] = matO[j][hidLength] - learningRate * deltaOut[j];
			}

		}
		//atualizacao do valor do erro medio
		erroMLP = erroMLP/qtTrainCases;
	}
	cout << "erro medio = " << erroMLP << endl;
}