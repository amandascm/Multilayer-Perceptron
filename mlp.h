#ifndef MLP_H
#define MLP_H


//DEFINES
#define inLength 2
#define hidLength 2
#define outLength 1
#define inFile "xor.txt"
#define learningRate 0.1
#define threshold 0.001

//GLOBAL
int qtTrainCases = 0; //quantidade de linhas do datase (casos teste para treinar o MLP)


//CLASSES
typedef struct model{
	float matH[hidLength][inLength+1]; //pesos+bias de cada neuronio da camada H em uma linha
	float matO[outLength][hidLength+1]; //pesos+bias de cada neuronio da camada O em uma linha
	float outResult[outLength]; //resultados obtidos em cada neuronio da camada O (apos aplicar funcao de ativacao)
	float hidResult[hidLength]; //resultados obtidos em cada neuronio da camada H (apos aplicar funcao de ativacao)

}model;


//FUNCTIONS BODIES

float activFunc(float z){
	//sigmoide
	return (1.0/(1.0 + expf(-z)));
}

float activFuncDeriv(float z){
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
	for(i=0;i<outLength;i++){
		for(j=0;j<(hidLength+1);j++){
			arch->matO[i][j] = 2.0f * ((float)rand() / (2.0f * (float)RAND_MAX)) - 0.5f;
		}
	}
}

void forward(float* inVector, model* arch){
	int i, j;
	float totalH = 0, totalO = 0;

	for(i=0;i<hidLength;i++){
		totalH = 0;
		for(j=0;j<(inLength);j++){
			totalH += arch->matH[i][j] * inVector[j]; // + w*x
		}
		 totalH += arch->matH[i][inLength]; // + bias
		 arch->hidResult[i] = activFunc(totalH);
	}

	for(i=0;i<outLength;i++){
		totalO = 0;
		for(j=0;j<hidLength;j++){
			totalO += arch->matO[i][j] * arch->hidResult[j]; // + w*z
		}
		totalO += arch->matO[i][hidLength]; // + bias
		arch->outResult[i] = activFunc(totalO);
	}
}

void backpropagation(float X[][inLength], float Y[][outLength], model* arch){
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
			forward(inVector, arch);

			//calculo do delta de cada neuronio da camada output
			for(j=0;j<outLength;j++){
				erro = Y[i][j] - arch->outResult[j];
				erroMLP += pow(erro, 2);
				deltaOut[j] = -2*erro*activFuncDeriv(arch->outResult[j]);
			}

			//calculo do delta de cada neuronio da camada hidden
			for(j=0;j<hidLength;j++){
				sum = 0;
				for(k=0;k<outLength;k++){
					sum += deltaOut[k] * arch->matO[k][j];
				}
				deltaHid[j] = sum * activFuncDeriv(arch->hidResult[j]);
			}

			//atualizacao de pesos e bias da camada hidden
			for(j=0;j<hidLength;j++){
				for(k=0;k<inLength;k++){
					arch->matH[j][k] = arch->matH[j][k] - learningRate * deltaHid[j] * inVector[k];
				}
				arch->matH[j][k] = arch->matH[j][k] - learningRate * deltaHid[j];
			}

			//atualizacao de pesos e bias da camada output
			for(j=0;j<outLength;j++){
				for(k=0;k<hidLength;k++){
					arch->matO[j][k] = arch->matO[j][k] - learningRate * deltaOut[j] * arch->hidResult[k];
				}
				arch->matO[j][hidLength] = arch->matO[j][hidLength] - learningRate * deltaOut[j];
			}

		}
		//atualizacao do valor do erro medio
		erroMLP = erroMLP/qtTrainCases;
	}
	printf("erro medio = %f\n", erroMLP);
}


#endif