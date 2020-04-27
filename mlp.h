#ifndef MLP_H
#define MLP_H


//DEFINES
#define inLength 2
#define hidLength 2
#define outputLength 1
#define inFile "xor.txt"
#define learningRate 0.1
#define threshold 0.001

//GLOBAL
int qtTrainCases = 0;


//CLASSES
typedef struct model{
	float matH[hidLength][inLength+1]; //pesos+bias de cada neuronio da camada H em uma linha
	float matO[outputLength][hidLength+1]; //pesos+bias de cada neuronio da camada O em uma linha
	float outputResult[outputLength];
	float hidResult[hidLength];

}model;


//FUNCTIONS BODIES

float funcActiv(float z){
	//sigmoide
	return (1.0/(1.0 + expf(-z)));
}

float derivFuncActiv(float z){
	//derivada sigmoide
	return (z*(1.0 - z));
}

void fillRandom(model* arch){
	//preenche matrizes de pesos e biases com numeros pseudoaleatorios entre -0.5 e 0.5
	int i, j;
	for(i=0;i<hidLength;i++){
		for(j=0;j<(inLength+1);j++){
			arch->matH[i][j] = 2.0f * ((float)rand() / (2.0f * (float)RAND_MAX)) - 0.5f;
		}
	}
	for(i=0;i<outputLength;i++){
		for(j=0;j<(hidLength+1);j++){
			arch->matO[i][j] = 2.0f * ((float)rand() / (2.0f * (float)RAND_MAX)) - 0.5f;
		}
	}
}

void forward(int* inVector, model* arch){
	int i, j;
	float totalH = 0;
	float totalO = 0;

	for(i=0;i<hidLength;i++){
		totalH = 0;
		for(j=0;j<(inLength);j++){
			totalH += arch->matH[i][j] * inVector[j]; // + w*x
		}
		 totalH += arch->matH[i][inLength]; // + bias
		 arch->hidResult[i] = funcActiv(totalH);
		 //printf("Z do neuronio[%d] camada hidden = %f\n", i, arch->hidResult[i]);
	}

	for(i=0;i<outputLength;i++){
		totalO = 0;
		for(j=0;j<hidLength;j++){
			totalO += arch->matO[i][j] * arch->hidResult[j]; // + w*z
		}
		totalO += arch->matO[i][hidLength];
		arch->outputResult[i] = funcActiv(totalO);
		//printf("Z do neuronio[%d] camada output = %f\n", i, arch->outputResult[i]);
	}
}

void backpropagation(int X[][inLength], int Y[][outputLength], model* arch){
	int i, j, k;
	int inVector[inLength];
	float Yo[qtTrainCases][outputLength]; //Vetor de saidas obtidas a ser comparado com o vetor Y de saidas esperadas
	float finalError = 2*threshold, error[outputLength];
	float deltaO[outputLength], deltaH[hidLength];
	float derivErrorWeightBiasOut[outputLength][hidLength+1];
	float derivErrorWeightBiasHid[hidLength][inLength+1];
	while(finalError > threshold){
		finalError = 0;

		//calculando os resultados obtidos para os casos teste antes do treinamento do MLP
		for(i=0;i<qtTrainCases;i++){

			for(j=0;j<inLength;j++){
				inVector[j] = X[i][j];
			}

			forward(inVector, arch);
			for(j=0;j<outputLength;j++){
				Yo[i][j] = arch->outputResult[j]; //Saidas obtidas
				error[j] = Y[i][j] - Yo[i][j]; //vetor com Y - Yo de cada caso teste/linha do dataset
				finalError += pow(error[j], 2); //somatorio de (Y - Yo)^2 = E
			}
			//deltaO = -2*(y - yo)*derivResultNeuronio
			for(j=0;j<outputLength;j++){
				deltaO[j] = -2.0 * error[j] * derivFuncActiv(arch->outputResult[j]);
			}
			//calculando derivada do Erro com relacao aos pesos e ao bias de cada neuronio da camada output
			for(j=0;j<outputLength;j++){
				for(k=0;k<hidLength;k++){
					// dE/dw(j,k) = deltaO(j)*resultH(k)
					derivErrorWeightBiasOut[j][k] = deltaO[j] * arch->hidResult[k];
				}
				// dE/db(j) = deltaO(j)
				derivErrorWeightBiasOut[j][hidLength] = deltaO[j]; //derivada do erro com relacao ao bias
			}

			//SOMENTE ATUALIZAR PESOS e BIAS APOS TODOS OS CALCULOS DE TODAS AS CAMADAS
			
			//deltaH = (somatorio(deltaO * pesosO)) * derivResultNeuronio
			for(j=0;j<hidLength;j++){
				deltaH[j] = 0;
				for(k=0;k<outputLength;k++){
					deltaH[j] += deltaO[k] * arch->matO[k][j];
				}
				deltaH[j] = deltaH[j] * derivFuncActiv(arch->hidResult[j]);
			}
			//calculando a derivada do Erro com relacao aos pesos e ao bias de cada neuronio da camada hidden
			for(j=0;j<hidLength;j++){
				for(k=0;k<inLength;k++){
					// dE/dw(j,k) = deltaH(j)*entrada(k)
					derivErrorWeightBiasHid[j][k] = deltaH[j] * inVector[k];
				}
				// dE/db(j) = deltaH(j)
				derivErrorWeightBiasHid[j][inLength] = deltaH[j];
			}

			// w(j,k) = w(j,k) - learningRate*derivErroWeight
			// b(j) = b(j) - learningRate*derivErroBias
			for(j=0;j<outputLength;j++){
				for(k=0;k<hidLength+1;k++){
					arch->matO[j][k] = arch->matO[j][k] - learningRate * derivErrorWeightBiasOut[j][k];
				}
			}

			for(j=0;j<hidLength;j++){
				for(k=0;k<inLength+1;k++){
					arch->matH[j][k] = arch->matH[j][k] - learningRate * derivErrorWeightBiasHid[j][k];
				}
			}


		}

		finalError = finalError/qtTrainCases; //erro medio
		//printf("erro medio = %f\n", finalError);
	}
	printf("erro medio = %f\n", finalError);
	printf("backpropagation finished\n");

}


#endif