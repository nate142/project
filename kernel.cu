#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "cudnn.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cstring>

#include <tchar.h>
#include <windows.h>
#include <SDKDDKVer.h>
#include <fstream>

#include <random>

#include <iostream>
#include <thread>
#include <mutex>

#define byte unsigned char
#define CenterInput false //nullifies the fmaxInput and fminInput
#define fmaxInput 1 //Images 0-255 values will be scaled to these input values automatically
#define fminInput -1 
#define fmaxOutput 1
#define fminOutput 0
#define lnOverflowValue log(1.192093e-07) //used in place of log(0)
#define TanhModifier 0.01f //notation: 1 or 0.1f NOT 1f
#define BaisActive
#define RandomGaussianOffset 0.0
#define RandomGaussianStdDeviation 0.5 //this needs to be changed.
#define TrainStatusUpdate 10000

#define TrainingDataCount 5 /*files*/ * 10000 /*images*/
#define TestDataCount 1 /*files*/ * 10000 /*images*/
#define ExampleInputDepth 3
#define ExampleInputWidth 32
#define ExampleInputHeight 32
#define ExampleOutputDepth 10 //possible outputs

#pragma region MagicNumberDefn
#define DefaultWorkSpaceSize 64800000 //in bytes
#pragma endregion

#pragma region GlobalSettings
float LearningRate = 0.001f;
float MomentumCoefficient = 0.00f; //zero means disabled
float RegressionCoefficient = 0.0f; //For each update: w' = (1 - LearningRate * RegressionCoefficient) * w - dJ/dw where J is cost, Set RegressionCoefficient to 0 to disable regression
float RegressionCoefficientMax = 1.0f;
float RegressionCoefficientMin = 0.0f;
float RegressionCoefficientEpsilon = 1e-22f;
#define ValidationSetSize 3000
size_t batchCount = 125;// 125;

//#define LocalResponseNormalization
#ifdef LocalResponseNormalization
float LocalResponseNormalizationAlpha = 0.00001f;
float LocalResponseNormalizationBeta = 1.0f;
int LocalResponseNormalizationNMin = -4; //sum from i + LocalResponseNormalizationNMin to i + LocalResponseNormalizationNMax
int LocalResponseNormalizationNMax = 5;
#endif
#pragma endregion


/*
Naming rules:
Capitals for first letter of global vars
Count = number of structs
Size = number of bytes

prefixes:
t = Type or anything like *_t
c = Array in GPU memory (Cuda)
h = Array in Host memory (Host) <- not required tho

*/

cudnnHandle_t lib;

#pragma region basic functions
size_t Max(size_t a, size_t b)
{
	if (a >= b)
		return a;
	else
		return b;
}
size_t Min(size_t a, size_t b)
{
	if (a <= b)
		return a;
	else
		return b;
}

void Pause()
{
#if defined(DestoryLibOnPause)
	cudnnDestroy(lib);
#endif
	getchar();
}
void OnExit()
{
	printf("Press any key to exit...");
	Pause();
}
void Try(cudnnStatus_t result)
{
	if (result != cudnnStatus_t::CUDNN_STATUS_SUCCESS)
	{
		printf("FATAL ERROR: ");
		printf(cudnnGetErrorString(result));
		Pause();
	}
}
void Try(cudaError_t result)
{
	if (result != cudaError_t::cudaSuccess)
	{
		printf("FATAL ERROR: cuda.");
		Pause();
	}
}
char * StrConcat(char* str1, char* str2)
{
	static  char* full_text;
	full_text = (char*)malloc(strlen(str1) + strlen(str2) + 1);
	strcpy(full_text, str1);
	strcat(full_text, str2);
	return full_text;
}
void Error(char* str1)
{
	printf(StrConcat("FATAL ERROR: ", str1));
	while (true)
	{
		getchar();
	}
}
void RewriteLine_MoveCursor()
{
	CONSOLE_SCREEN_BUFFER_INFO csbi;
	COORD  result;
	GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &csbi);
	result.X = 0;
	result.Y = csbi.dwCursorPosition.Y;
	SetConsoleCursorPosition(GetStdHandle(STD_OUTPUT_HANDLE), result);
}


WORD ConsoleColor_saved_attributes;
HANDLE ConsoleColor_ConsoleHandle;
void ConsoleColorSetup()
{
	ConsoleColor_ConsoleHandle = GetStdHandle(STD_OUTPUT_HANDLE);
	CONSOLE_SCREEN_BUFFER_INFO consoleInfo;
	GetConsoleScreenBufferInfo(ConsoleColor_ConsoleHandle, &consoleInfo);
	ConsoleColor_saved_attributes = consoleInfo.wAttributes;
}
void ConsoleColorReset()
{
	SetConsoleTextAttribute(ConsoleColor_ConsoleHandle, ConsoleColor_saved_attributes);
}
void ConsoleColorGreen()
{
	SetConsoleTextAttribute(ConsoleColor_ConsoleHandle, FOREGROUND_GREEN | FOREGROUND_INTENSITY);
}
void ConsoleColorRed()
{
	SetConsoleTextAttribute(ConsoleColor_ConsoleHandle, FOREGROUND_RED | FOREGROUND_INTENSITY);
}
void ConsoleColorBlue()
{
	SetConsoleTextAttribute(ConsoleColor_ConsoleHandle, FOREGROUND_BLUE | FOREGROUND_INTENSITY);
}


#pragma endregion

#pragma region MultiThreading
std::mutex ThreadSyncMutex;
bool FirstWeightsReadyFlag;
float** FirstWeightArray;
int FirstWeightIndex;
#pragma endregion

#pragma region math functions
std::random_device rd;
std::default_random_engine randomEngine(rd());
std::normal_distribution<double> distribution(RandomGaussianOffset, RandomGaussianStdDeviation);
float RandomGassian()
{
	return distribution(randomEngine);
}
double RandomDouble()
{
	double f = (double)rand() / RAND_MAX;
	return 0.0 + f * (1.0 - 0.0);
}
#define StoredRandomSeedCount 1024
size_t StoredRandomSeedIndex = StoredRandomSeedCount;
#define seed unsigned long long
seed* StoredRandomSeeds = (seed*)malloc(sizeof(seed) * StoredRandomSeedCount);
//NOT Multithread Safe
seed RandomSeed() 
{
	//ThreadSyncMutex.lock();
	if (StoredRandomSeedIndex >= StoredRandomSeedCount)
	{
		if (RAND_MAX != 0x7fff)
		{
			Error("RAND_MAX != 0x7fff");
		}
		StoredRandomSeedIndex = 0;
		for (size_t i = 0; i < StoredRandomSeedCount; i++)
		{
			seed newSeed = 0;
			for (int i = 0; i<64; i += 30) {
				newSeed = newSeed*(RAND_MAX + (seed)1) + rand();
			}
			StoredRandomSeeds[i] = newSeed;
		}
	}
	seed sed = StoredRandomSeeds[StoredRandomSeedIndex];
	StoredRandomSeedIndex++;
	//ThreadSyncMutex.unlock();
	return sed;
}
//NOT Multithread Safe
float GetRandomWeight()  
{
	return RandomGassian() * 0.01f;
}
int OutputSize(int input_width, int pad_w, int receptorPoolWidth, int receptorPoolStride)
{
	int WidthTotal = input_width + (2 * pad_w);
	double output_width = ((WidthTotal - receptorPoolWidth) / receptorPoolStride) + 1;
	if (output_width < 1 || output_width != (int)output_width)
	{//either strideX is larger than pool width, or pool does not divide nicely
		Error("Invalid pool size parameters.");
	}
	return (int)output_width;
}
#pragma endregion

#pragma region depericated functions
byte* ConvertNCHW_to_NHWC(byte* data)
{
	size_t counter = 0;
	byte* imageBufferOut = (byte*)malloc(sizeof(byte) * 32*32*3);
	for (size_t y = 0; y < 32; y++)
	{
		for (size_t x = 0; x < 32; x++)
		{
			for (size_t c = 0; c < 3; c++)
			{
				size_t location = (c * 1024) + (y * 32) + x;
				imageBufferOut[counter] = data[location];
				counter++;
			}
		}
	}
	return imageBufferOut;
}
#pragma endregion

#pragma region GlobalPointers_no_settings
cudnnConvolutionFwdAlgo_t AlgoFeedFoward = cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_GEMM;
cudnnConvolutionBwdDataAlgo_t AlgoBackData = cudnnConvolutionBwdDataAlgo_t::CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;
cudnnConvolutionBwdFilterAlgo_t AlgoBackFilter = cudnnConvolutionBwdFilterAlgo_t::CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0;

cudnnActivationDescriptor_t ActivationFunctionTanh;
cudnnActivationDescriptor_t ActivationFunctionReLu;

float* ImageData;
float* LabelData;

float PointerZero = 0; //used for when momentum will never be required
float PointerOne = 1;
float PointerMinusOne = -1;
bool DropoutMergedNetwork = false;
size_t BatchTrainingEpochs;
#pragma endregion

#pragma region UtillyFunctions
bool isNaN(float val)
{
	return val != val || (!(val >= 0) && !(val < 0));
}
bool FindNaNs(float* hData, size_t count, char* description, int LayerIndex)
{
	for (size_t i = 0; i < count; i++)
	{
		if (isNaN(hData[i]))
		{
			char* text = StrConcat("Alert! NaN at: ", StrConcat(description, " LayerIndex=%i, i=%i\nPaused...\n"));
			printf(text, LayerIndex, i);
			Pause();
			return true;
		}
	}
	return false;
}

#pragma endregion



enum layerType { FeedFowardActivation, FeedFowardSoftmax, Example, Convolutional, Pooling, Dropout };
struct Layer
{
	float* CommonWorkspace;
	void* DropoutRandomState;
	void* DropoutReserveSpace;
	size_t DropoutRandomStateSize;
	size_t DropoutReserveSpaceSize;
	float DropoutChance;
	seed DropoutSeed;

	cudnnTensorDescriptor_t tNeurons;
	cudnnFilterDescriptor_t tWeights;
	cudnnTensorDescriptor_t tWeightTensor;
	cudnnTensorDescriptor_t tBias;
	cudnnDropoutDescriptor_t tDropout;
	layerType LayerType;
	float* cNeurons;
	float* cNeuronsErrors;
#ifdef BaisActive
	float* cBias;
	float* cBiasVelocity;
#endif
	float* cWeights;
	float* cWeightsVelocity;
#ifdef LocalResponseNormalization
	float* cGamma; //review, bad name
	cudnnOpTensorDescriptor_t tNeuronOpMultiply;
	bool LRN = true;
#endif

	size_t Width;
	size_t Height;
	size_t Depth;

	size_t InputWidth;
	size_t InputHeight;
	size_t InputDepth;

	size_t BatchCount;

	cudnnActivationDescriptor_t tActivationFunction = NULL;

private: 
	cudnnConvolutionDescriptor_t tConvoMode;
	cudnnPoolingDescriptor_t tPooling;
	bool workspaceBigEnough = false;
	int StrideY = 1;
	int StrideX = 1;
	int InputPaddingHeight = 0;
	int InputPaddingWidth = 0;

	#pragma region Private Example vars
	size_t TotalExampleCount;
	float* cExamples;
	
	#pragma endregion
	void init()
	{
		Try(cudnnCreateTensorDescriptor(&tNeurons));
		Try(cudnnSetTensor4dDescriptor(tNeurons, cudnnTensorFormat_t::CUDNN_TENSOR_NCHW, cudnnDataType_t::CUDNN_DATA_FLOAT, BatchCount, Depth, Height, Width));
		Try(cudaMalloc(&cNeurons, NeuronSizeMemory()));
		Try(cudaMalloc(&cNeuronsErrors, NeuronSizeMemory()));

	    if (LayerType != layerType::Example && LayerType != layerType::Pooling && LayerType != layerType::Dropout)
		{
#ifdef BaisActive
			Try(cudnnCreateTensorDescriptor(&tBias));
			Try(cudnnSetTensor4dDescriptor(tBias, cudnnTensorFormat_t::CUDNN_TENSOR_NCHW, cudnnDataType_t::CUDNN_DATA_FLOAT, 1/*bias is shared like weights*/, Depth, 1/*each filter has only a single bias*/, 1/*each filter has only a single bias*/));
			Try(cudaMalloc(&cBias, BiasSizeMemory()));
			Try(cudaMalloc(&cBiasVelocity, BiasSizeMemory()));
			Try(cudaMemset(cBiasVelocity, 0x0, BiasSizeMemory()));
#endif

			Try(cudnnCreateFilterDescriptor(&tWeights));
			Try(cudnnSetFilter4dDescriptor(tWeights, cudnnDataType_t::CUDNN_DATA_FLOAT, cudnnTensorFormat_t::CUDNN_TENSOR_NCHW, Depth, InputDepth, InputHeight, InputWidth));
			Try(cudnnCreateTensorDescriptor(&tWeightTensor));
			Try(cudnnSetTensor4dDescriptor(tWeightTensor, cudnnTensorFormat_t::CUDNN_TENSOR_NCHW, cudnnDataType_t::CUDNN_DATA_FLOAT, Depth, InputDepth, InputHeight, InputWidth));
			Try(cudaMalloc(&cWeights, WeightSizeMemory()));
			Try(cudaMalloc(&cWeightsVelocity, WeightSizeMemory()));
			Try(cudaMemset(cWeightsVelocity, 0x0, WeightSizeMemory()));

			Try(cudnnCreateConvolutionDescriptor(&tConvoMode));
			Try(cudnnSetConvolution2dDescriptor(tConvoMode, InputPaddingHeight, InputPaddingWidth, StrideY, StrideX, 1, 1, cudnnConvolutionMode_t::CUDNN_CONVOLUTION));

			RandomiseWeights();
		}
	}
public:
	void RandomiseWeights()
	{
		if (LayerType != layerType::Example && LayerType != layerType::Pooling && LayerType != layerType::Dropout)
		{
			float* randomWeights = (float*)malloc(WeightSizeMemory());
			size_t cnt = WeightCountMemory();
			for (size_t i = 0; i < cnt; i++)
			{
				randomWeights[i] = GetRandomWeight();
			}
			Try(cudaMemcpy(cWeights, randomWeights, WeightSizeMemory(), cudaMemcpyKind::cudaMemcpyHostToDevice));
			free(randomWeights);
#ifdef BaisActive
			randomWeights = (float*)malloc(BiasSizeMemory());
			cnt = BiasCountMemory();
			for (size_t i = 0; i < cnt; i++)
			{
				randomWeights[i] = GetRandomWeight();
			}
			Try(cudaMemcpy(cBias, randomWeights, BiasSizeMemory(), cudaMemcpyKind::cudaMemcpyHostToDevice));
			free(randomWeights);
#endif
		}
	}
	Layer()
	{}
	void ConstructFullyConnectedLayer_ActivationFunction(float* commonWorkspace, size_t batchCount, Layer input, size_t depth, cudnnActivationDescriptor_t activationFunction)
	{
		CommonWorkspace = commonWorkspace;

		BatchCount = batchCount;
		Depth = depth;
		Height = 1;
		Width = 1;
		InputDepth = input.Depth;
		InputHeight = input.Height;
		InputWidth = input.Width;
		tActivationFunction = activationFunction;
		LayerType = layerType::FeedFowardActivation;
		init();
		printf("Added Fully Connected Activation Function Layer: Depth=%i\n", Depth);
	}
	void ConstructFullyConnectedLayer_Softmax(float* commonWorkspace, size_t batchCount, Layer input, size_t depth)
	{
		CommonWorkspace = commonWorkspace;

		BatchCount = batchCount;
		Depth = depth;
		Height = 1;
		Width = 1;
		InputDepth = input.Depth;
		InputHeight = input.Height;
		InputWidth = input.Width;
		tActivationFunction = NULL; //this is bloody important
		LayerType = layerType::FeedFowardSoftmax;
		init();
		printf("Added Fully Connected Softmax Layer: Depth=%i\n", Depth);
	}
	void ConstructConvolutionalLayer(float* commonWorkspace, size_t batchCount, Layer input, size_t depth, size_t receptorHeight, size_t receptorWidth, cudnnActivationDescriptor_t activationFunction)
	{
		ConstructConvolutionalLayer(commonWorkspace, batchCount, input, depth, receptorHeight, receptorWidth, activationFunction, 1, 1, 0, 0);
	}
	void ConstructConvolutionalLayer(float* commonWorkspace, size_t batchCount, Layer input, size_t depth, size_t receptorHeight, size_t receptorWidth, cudnnActivationDescriptor_t activationFunction, int strideY, int strideX, int inputPaddingHeight, int inputPaddingWidth)
	{
		CommonWorkspace = commonWorkspace;

		BatchCount = batchCount;
		Depth = depth;
		InputDepth = input.Depth;
		InputHeight = receptorHeight;
		InputWidth = receptorWidth;

		InputPaddingHeight = inputPaddingHeight;
		InputPaddingWidth = inputPaddingWidth;
		StrideX = strideX;
		StrideY = strideY;

		Width = OutputSize(input.Width, InputPaddingWidth, receptorWidth, strideX);
		Height = OutputSize(input.Height, InputPaddingHeight, receptorHeight, strideY);

		tActivationFunction = activationFunction;
		LayerType = layerType::Convolutional;

		init();
		printf("Added Convolution Layer: Depth=%i, Height=%i, Width=%i, ReceptorHeight=%i, ReceptorWidth=%i, StrideY=%i, StrideX=%i\n",
			Depth, Height, Width, receptorHeight, receptorWidth, StrideY, StrideX);
	}
	void ConstructPoolingLayer(float* commonWorkspace, size_t batchCount, Layer input, size_t poolHeight, size_t poolWidth, cudnnPoolingMode_t poolingMode)
	{
		ConstructPoolingLayer(commonWorkspace, batchCount, input, poolHeight, poolWidth, poolingMode, poolHeight, poolWidth, 0, 0);
	}
	void ConstructPoolingLayer(float* commonWorkspace, size_t batchCount, Layer input, size_t receptorPoolHeight, size_t receptorPoolWidth, cudnnPoolingMode_t poolingMode, int strideY, int strideX, int inputPaddingHeight, int inputPaddingWidth)
	{
		CommonWorkspace = commonWorkspace;

		BatchCount = batchCount;
		Depth = input.Depth;

		InputPaddingHeight = inputPaddingHeight;
		InputPaddingWidth = inputPaddingWidth;
		StrideX = strideX;
		StrideY = strideY;
		
		Width = OutputSize(input.Width, InputPaddingWidth, receptorPoolWidth, strideX);
		Height = OutputSize(input.Height, InputPaddingHeight, receptorPoolHeight, strideY);

		LayerType = layerType::Pooling;

		Try(cudnnCreatePoolingDescriptor(&tPooling));
		Try(cudnnSetPooling2dDescriptor(tPooling, poolingMode, cudnnNanPropagation_t::CUDNN_PROPAGATE_NAN, receptorPoolHeight, receptorPoolWidth, InputPaddingHeight, InputPaddingWidth, StrideY, StrideX));
		tActivationFunction = NULL;
		init();

#ifdef LocalResponseNormalization
		if (LRN || true)
		{
			Try(cudaMalloc(&cGamma, NeuronSizeMemory()));
			Try(cudnnCreateOpTensorDescriptor(&tNeuronOpMultiply));
			Try(cudnnSetOpTensorDescriptor(tNeuronOpMultiply, cudnnOpTensorOp_t::CUDNN_OP_TENSOR_MUL, cudnnDataType_t::CUDNN_DATA_FLOAT, cudnnNanPropagation_t::CUDNN_PROPAGATE_NAN));
		}
#endif

		printf("Added Pooling Layer: Depth=%i, Height=%i, Width=%i, ReceptorPoolHeight=%i, ReceptorPoolWidth=%i, StrideY=%i, StrideX=%i\n",
			Depth, Height, Width, receptorPoolHeight, receptorPoolWidth, StrideY, StrideX);
	}
	void ConstructExampleLayer(float* commonWorkspace, size_t batchCountPerTrain, size_t totalExampleCount, size_t depth, size_t height, size_t width, float* hExampleData)
	{
		CommonWorkspace = commonWorkspace;

		BatchCount = batchCountPerTrain;
		TotalExampleCount = totalExampleCount;
		if (TotalExampleCount % BatchCount != 0)
			Error("TotalExampleCount %% BatchCount != 0");
		Depth = depth;
		Height = height;
		Width = width;

		LayerType = layerType::Example;
		init();
		Try(cudaMalloc(&cExamples, ExampleSizeMemory()));
		Try(cudaMemcpy(cExamples, hExampleData, ExampleSizeMemory(), cudaMemcpyKind::cudaMemcpyHostToDevice));
		printf("Added Example Layer: Depth=%i, Height=%i, Width=%i\n",
			Depth, Height, Width);
	}
	void ConstructDropoutLayer(float* commonWorkspace, size_t batchCount, Layer input, float dropoutChance)
	{
		LayerType = layerType::Dropout;
		DropoutChance = dropoutChance;
		CommonWorkspace = commonWorkspace;

		BatchCount = batchCount;
		Width = input.Width;
		Height = input.Height;
		Depth = input.Depth;

		init();

		RandomiseDropoutNodes();
		Try(cudnnDropoutGetStatesSize(lib, &DropoutRandomStateSize));
		Try(cudaMalloc(&DropoutRandomState, DropoutRandomStateSize));
		Try(cudnnDropoutGetReserveSpaceSize(tNeurons, &DropoutReserveSpaceSize));
		Try(cudaMalloc(&DropoutReserveSpace, DropoutReserveSpaceSize));
		Try(cudnnCreateDropoutDescriptor(&tDropout));
	}
    void RandomiseDropoutNodes()
	{
		if (LayerType == layerType::Dropout)
			DropoutSeed = RandomSeed();
	}

	size_t ExampleCountMemory()
	{
		return TotalExampleCount * Depth * Height * Width;
	}
	size_t ExampleSizeMemory()
	{
		return ExampleCountMemory() * sizeof(float);
	}
	size_t NeuronCountMemory()
	{
		return BatchCount * Depth * Height * Width;
	}
	size_t BiasCountMemory()
	{
		return Depth * Height * Width;
	}
	size_t WeightCountMemory()
	{
		return Depth * InputDepth * InputHeight * InputWidth;
	}
	size_t NeuronSizeMemory()
	{
		return NeuronCountMemory() * sizeof(float);
	}
	size_t BiasSizeMemory()
	{
		return BiasCountMemory() * sizeof(float);
	}
	size_t WeightSizeMemory()
	{
		return WeightCountMemory() * sizeof(float);
	}
    
	void FeedFoward(int startIndex)
	{
		if (LayerType != layerType::Example)
			Error("Wrong ff called.");
		cNeurons = &cExamples[startIndex * Depth * Height * Width];
	}
	void FeedFoward(Layer input)
	{
		if (LayerType == layerType::Example)
			Error("Wrong ff called for example layer.");
		if (!workspaceBigEnough)
		{
			size_t outSize = 0;//dont know if this =0 is needed
			if (LayerType != layerType::Pooling)
			{
				size_t outSizeAlt1 = 0;
				size_t outSizeAlt2 = 0;
				if (LayerType != layerType::Dropout)
				{
					Try(cudnnGetConvolutionForwardWorkspaceSize(lib, input.tNeurons, tWeights, tConvoMode, tNeurons, AlgoFeedFoward, &outSize));
					Try(cudnnGetConvolutionBackwardDataWorkspaceSize(lib, tWeights, tNeurons, tConvoMode, input.tNeurons, AlgoBackData, &outSizeAlt1));
					Try(cudnnGetConvolutionBackwardFilterWorkspaceSize(lib, input.tNeurons, tNeurons, tConvoMode, tWeights, AlgoBackFilter, &outSizeAlt2));
					outSize = Max(Max(outSizeAlt2, outSizeAlt1), outSize);
				}
				if (outSize > DefaultWorkSpaceSize)
				{//increase size
					printf("ERROR: Workspace is too small. Set it to: %i", outSize);
					Error("");
				}
			}
			workspaceBigEnough = true;
		}

		if (LayerType == layerType::Pooling)
		{
			Try(cudnnPoolingForward(lib, tPooling, &PointerOne, input.tNeurons, input.cNeurons, &PointerZero, tNeurons, cNeurons));

#ifdef LocalResponseNormalization
			if (LRN)
			{
				float* activations = (float*)malloc(NeuronSizeMemory());
				float* newActivations = (float*)malloc(NeuronSizeMemory());
				float* newGamma = (float*)malloc(NeuronSizeMemory());
				GetNeuronData(activations);
				int DistanceBetweenFilters = Height * Width;
				for (int exampleI = 0; exampleI < BatchCount; exampleI++)
				{
					for (int z = 0; z < Depth; z++)
					{
						for (int y = 0; y < Height; y++)
						{
							for (int x = 0; x < Width; x++)
							{
								float sum = 0;
								float activation;
								for (int i = z + LocalResponseNormalizationNMin; i < z + LocalResponseNormalizationNMax; i++)
								{
									int myI = i % Depth;
									activation = activations[(((((exampleI * Depth) + myI) * Height) + y) * Width) + x];
									sum += activation * activation;
								}
								int currentIndex = (((((exampleI * Depth) + z) * Height) + y) * Width) + x;
								newGamma[currentIndex] = pow((1.0f + (LocalResponseNormalizationAlpha * sum)), -LocalResponseNormalizationBeta);
								newActivations[currentIndex] = activations[currentIndex] * newGamma[currentIndex];
							}
						}
					}
				}
				SetNeuronData(newActivations);
				cudaMemcpy(cGamma, newGamma, NeuronSizeMemory(), cudaMemcpyKind::cudaMemcpyHostToDevice);
				free(newActivations);
				free(activations);
				free(newGamma);
			}
#endif
		}
		else if (LayerType == layerType::Dropout)
		{
			if (DropoutMergedNetwork)
			{
				float* oldcNeurons = cNeurons;
				cNeurons = input.cNeurons;
				input.cNeurons = cNeurons;
			}
			else
			{
				Try(cudnnSetDropoutDescriptor(tDropout, lib, DropoutChance, DropoutRandomState, DropoutRandomStateSize, DropoutSeed));
				Try(cudnnDropoutForward(lib, tDropout, input.tNeurons, input.cNeurons, tNeurons, cNeurons, DropoutReserveSpace, DropoutReserveSpaceSize));
			}
		}
		else
		{
			float alpha = 1;
			if (tActivationFunction == ActivationFunctionTanh)
			{
				alpha *= TanhModifier;//to scale down sum output for tanh
			}

			Try(cudnnConvolutionForward(lib, &alpha, input.tNeurons, input.cNeurons, tWeights, cWeights, tConvoMode, AlgoFeedFoward, CommonWorkspace,
				DefaultWorkSpaceSize, &PointerZero, tNeurons, cNeurons)); //invalid value error means workspace is too small

#ifdef BaisActive
			Try(cudnnAddTensor(lib, &PointerOne, tBias, cBias, &PointerOne, tNeurons, cNeurons));
#endif

			if (tActivationFunction != NULL)
			{
				Try(cudnnActivationForward(lib, tActivationFunction, &PointerOne, tNeurons, cNeurons, &PointerZero, tNeurons, cNeurons));
			}
			else
			{//possible softmax
				if (LayerType == layerType::FeedFowardSoftmax)
				{
					Try(cudnnSoftmaxForward(lib, cudnnSoftmaxAlgorithm_t::CUDNN_SOFTMAX_ACCURATE, cudnnSoftmaxMode_t::CUDNN_SOFTMAX_MODE_INSTANCE, &PointerOne, tNeurons, cNeurons, &PointerZero, tNeurons, cNeurons));
				}
			}



		}
	}
	void Backpropagate(Layer input, int startIndex)
	{
		if (LayerType != layerType::Example)
			Error("Wrong bp called.");

		//float* desired = &cExamples[startIndex * Depth * Height * Width];
		Try(cudnnAddTensor(lib, &PointerOne, input.tNeurons, input.cNeurons, &PointerZero, input.tNeurons, input.cNeuronsErrors));
		Try(cudnnAddTensor(lib, &PointerMinusOne, tNeurons, /*desired*/cNeurons, &PointerOne, input.tNeurons, input.cNeuronsErrors)); //delta = -(e_correct - e) = e - ecorrect = -1.ecorrect + 1.e
	}
	float CrossEntropyError(Layer input) //slow cpu function
	{
		if (fminOutput != 0 || fmaxOutput != 1)
			Error("Cross entropy error can only be used when fminOutput=0 and fmaxOutput=1, because of the properties of the ln(x) function.");
		if (input.LayerType != layerType::FeedFowardSoftmax)
			Error("Cross entropy error can only be used for FeedFowardSoftmax.");
	
		float* hNeurons = (float*)malloc(input.NeuronSizeMemory());
		input.GetNeuronData(hNeurons);

		float* hNeuronsDesired = (float*)malloc(NeuronSizeMemory());
		GetNeuronData(hNeuronsDesired);

		float sum = 0;
		int counter = 0;
		for (size_t n = 0; n < BatchCount; n++)
		{
			for (size_t d = 0; d < Depth; d++)
			{
				{
					{
						if (hNeuronsDesired[counter] == fmaxOutput)
						{
							if (hNeurons[counter] == 0)
								sum += lnOverflowValue;
							else
								sum += log(hNeurons[counter]);

#if defined(NaNCheck)
							if (isNaN(sum))
							{
								Error("NaN");
							}
#endif
						}
						
						counter++;
					}
				}
			}
		}
		free(hNeurons);
		return -sum;
	}
	float MeanSquaredError(Layer input, int startIndex) //slow cpu function
	{
		if (input.LayerType == layerType::FeedFowardSoftmax)
			Error("FeedFowardSoftmax can only use Cross entropy error");
		Backpropagate(input, startIndex);
		float* hNeurons = (float*)malloc(input.NeuronSizeMemory());
		input.GetNeuronErrorData(hNeurons);
		float squaredSum = 0;
		int counter = 0;
		for (size_t n = 0; n < BatchCount; n++)
		{
			for (size_t d = 0; d < Depth; d++)
			{
				for (size_t h = 0; h < Height; h++)
				{
					for (size_t w = 0; w < Width; w++)
					{
						squaredSum += hNeurons[counter] * hNeurons[counter];
						counter++;
					}
				}
			}
		}
		free(hNeurons);
		return ((squaredSum));
	}
	int CorrectExampleCount(Layer input) //slow cpu function
	{
		float* hNeurons = (float*)malloc(input.NeuronSizeMemory());
		input.GetNeuronData(hNeurons);
		float* hDesired = (float*)malloc(NeuronSizeMemory());
		GetNeuronData(hDesired);
		int CorrectSum = 0;
		int counter = 0;
		for (size_t n = 0; n < BatchCount; n++)
		{
			float MaxVal = fminOutput;
			int localMax = -2; //-1 = no/invalid index
			int localDesiredMax = -1; //-1 = no/invalid index
			for (size_t d = 0; d < Depth; d++)
			{
				for (size_t h = 0; h < Height; h++)
				{
					for (size_t w = 0; w < Width; w++)
					{
						if (hNeurons[counter] > MaxVal)
						{
							MaxVal = hNeurons[counter];
							localMax = counter;
						}
						if (hDesired[counter] == fmaxOutput)
						{
							if (localDesiredMax != -1)
								Error("Multiple fmax values in desired output of a single example.");
							localDesiredMax = counter;
						}
						counter++;
					}
				}
			}
			if (localMax == localDesiredMax)
				CorrectSum++;
		}
		free(hNeurons);
		free(hDesired);
		return CorrectSum;
	}

	void ComboErrorCount(Layer input, int* CorrectSum, float* CrossEntropySum) //slow cpu function
	{
		float* hNeurons = (float*)malloc(input.NeuronSizeMemory());
		input.GetNeuronData(hNeurons);
		float* hDesired = (float*)malloc(NeuronSizeMemory());
		GetNeuronData(hDesired);

		if (fminOutput != 0 || fmaxOutput != 1)
			Error("Cross entropy error can only be used when fminOutput=0 and fmaxOutput=1, because of the properties of the ln(x) function.");
		if (input.LayerType != layerType::FeedFowardSoftmax)
			Error("Cross entropy error can only be used for FeedFowardSoftmax.");

		int counter = 0;
		for (size_t n = 0; n < BatchCount; n++)
		{
			float MaxVal = fminOutput;
			int localMax = -2; //-1 = no/invalid index
			int localDesiredMax = -1; //-1 = no/invalid index
			for (size_t d = 0; d < Depth; d++)
			{
				{
					{
						if (hNeurons[counter] > MaxVal)
						{
							MaxVal = hNeurons[counter];
							localMax = counter;
						}
						if (hDesired[counter] == fmaxOutput)
						{
							if (localDesiredMax != -1)
								Error("Multiple fmax values in desired output of a single example.");
							localDesiredMax = counter;

							if (hNeurons[counter] == 0)
								*CrossEntropySum -= lnOverflowValue;
							else
								*CrossEntropySum -= log(hNeurons[counter]);
						}
#if defined(NaNCheck)
						if (isNaN(*CrossEntropySum))
						{
							Error("NaN");
						}
#endif
						counter++;
					}
				}
			}
			if (localMax == localDesiredMax)
				*CorrectSum += PointerOne;
		}
		free(hNeurons);
		free(hDesired);
	}

	void Backpropagate(Layer input)
	{//My Neurons have already been set to their delta
		if (LayerType == layerType::Example)
			Error("Wrong bp called for example layer.");

		if (tActivationFunction != NULL)
		{
			Try(cudnnActivationBackward(lib, tActivationFunction, &PointerOne, tNeurons, cNeurons/*f(sum) output: 1-f*f*/, tNeurons, cNeuronsErrors /*input delta -(ec-e)*/,
				tNeurons, cNeurons/*not used*/, &PointerZero, tNeurons, cNeuronsErrors/*output delta*/));
			//warning here mentioned above
			if (tActivationFunction != ActivationFunctionTanh && tActivationFunction != ActivationFunctionReLu)
				Error("Activation function backwards may not supported. See const void x* passed the cNeurons which it should not be! Also find same error below.");
		}

		//pass on neuron errors to child
		if (input.LayerType != layerType::Example)
		{
			if (LayerType == layerType::Pooling)
			{
#ifdef LocalResponseNormalization
				if (LRN)
				{//review, make this one line and test it?
					if (DefaultWorkSpaceSize < NeuronSizeMemory())
					{
						Error("CommonWorkspace too small.");
					}

					Try(cudnnOpTensor(lib, tNeuronOpMultiply, &PointerOne, tNeurons, cNeuronsErrors, &PointerOne, tNeurons, cGamma, &PointerZero, tNeurons, CommonWorkspace));
					Try(cudaMemcpy(cNeuronsErrors, CommonWorkspace, NeuronSizeMemory(), cudaMemcpyKind::cudaMemcpyDeviceToDevice)); //review, pointer swaping is faster
		
				}
#endif

				Try(cudnnPoolingBackward(lib, tPooling, &PointerOne, tNeurons, cNeurons, tNeurons, cNeuronsErrors, input.tNeurons, input.cNeurons, &PointerZero, input.tNeurons, input.cNeuronsErrors));
			}
			else if (LayerType == layerType::Dropout)
			{
				if (DropoutMergedNetwork)
				{
					float* oldcNeurons = cNeurons;
					cNeurons = input.cNeurons;
					input.cNeurons = cNeurons;

					oldcNeurons = cNeuronsErrors;
					cNeuronsErrors = input.cNeuronsErrors;
					input.cNeuronsErrors = cNeuronsErrors;
				}
				else
				{
					/*float* oldcNeurons = cNeurons;
					cNeurons = input.cNeurons;
					input.cNeurons = cNeurons;*/

					/*oldcNeurons = cNeuronsErrors;
					cNeuronsErrors = input.cNeuronsErrors;
					input.cNeuronsErrors = cNeuronsErrors;*/


				/*	input.PrintLayer(1);
					PrintLayer(1);
					input.PrintLayerError(1);
					PrintLayerError(1);*/

					//Try(cudaMemset(input.cNeuronsErrors, 0b0, input.NeuronSizeMemory()));
					//Try(cudaMemset(input.cNeurons, 0b0, input.NeuronSizeMemory()));

					//Try(cudaMemcpy(input.cNeurons, cNeurons, input.NeuronSizeMemory() / 2, cudaMemcpyKind::cudaMemcpyDeviceToDevice));
					//Try(cudaMemcpy(input.cNeuronsErrors, cNeuronsErrors, input.NeuronSizeMemory() / 2, cudaMemcpyKind::cudaMemcpyDeviceToDevice));
					
					/*printf("\n\n\ninput.PrintLayer\n");
					input.PrintLayer(0);
					printf("\n\n\nPrintLayer\n");
					PrintLayer(0);
					printf("\n\n\ninput.PrintLayerError\n");
					input.PrintLayerError(0);
					printf("\n\n\nPrintLayerError\n");
					PrintLayerError(0);
					Pause();*/

					//float reduction = 0.5f;
					//Try(cudnnAddTensor(lib, &PointerZero, tNeurons, cNeurons, &reduction, tNeurons, cNeurons));
					Try(cudnnDropoutBackward(lib, tDropout, tNeurons, cNeuronsErrors, input.tNeurons, input.cNeuronsErrors, DropoutReserveSpace, DropoutReserveSpaceSize));
				}
			}
			else
			{
				Try(cudnnConvolutionBackwardData(lib, &PointerOne, tWeights, cWeights, tNeurons, cNeuronsErrors, tConvoMode, AlgoBackData, CommonWorkspace,
					DefaultWorkSpaceSize, &PointerZero, input.tNeurons, input.cNeuronsErrors)); //invalid value error means workspace is too small
			}

			
		}

		if (LayerType != layerType::Pooling && LayerType != layerType::Dropout)
		{//momentum - http://ufldl.stanford.edu/tutorial/supervised/OptimizationStochasticGradientDescent/ where lambda is MomentumCoefficient and alpha is LearningRate
			//calc and modify my own weights
			Try(cudnnConvolutionBackwardFilter(lib, &LearningRate, input.tNeurons, input.cNeurons, tNeurons, cNeuronsErrors, tConvoMode, AlgoBackFilter, CommonWorkspace,
				DefaultWorkSpaceSize, &MomentumCoefficient, tWeights, cWeightsVelocity)); //invalid value error means workspace is too small

			float Beta = 1.0f - (LearningRate * RegressionCoefficient);

			Try(cudnnAddTensor(lib, &PointerMinusOne, tWeightTensor, cWeightsVelocity, &Beta, tWeightTensor, cWeights));

			//calc and modify my own bias
#ifdef BaisActive
			Try(cudnnConvolutionBackwardBias(lib, &LearningRate, tNeurons, cNeuronsErrors, &MomentumCoefficient, tBias, cBiasVelocity));

			Try(cudnnAddTensor(lib, &PointerMinusOne, tBias, cBiasVelocity, &Beta, tBias, cBias));
#endif
		}
	}

	void GetNeuronData(float* host)
	{
		Try(cudaMemcpy(host, cNeurons, NeuronSizeMemory(), cudaMemcpyKind::cudaMemcpyDeviceToHost));
	}
	void SetNeuronData(float* host)
	{
		Try(cudaMemcpy(cNeurons, host, NeuronSizeMemory(), cudaMemcpyKind::cudaMemcpyHostToDevice));
	}
	void GetNeuronErrorData(float* host)
	{
		Try(cudaMemcpy(host, cNeuronsErrors, NeuronSizeMemory(), cudaMemcpyKind::cudaMemcpyDeviceToHost));
	}
	void SetWeightData(float* host)
	{
		Try(cudaMemcpy(cWeights, host, WeightSizeMemory(), cudaMemcpyKind::cudaMemcpyHostToDevice));
	}
	void SetBiasData(float* host)
	{
#ifdef BaisActive
		Try(cudaMemcpy(cBias, host, BiasSizeMemory(), cudaMemcpyKind::cudaMemcpyHostToDevice));
#endif
	}
	void PrintLayer()
	{
		PrintLayer(-1);
	}
	void PrintLayerError()
	{
		PrintLayerError(-1);
	}
	void PrintLayerError(int earlyBreak)
	{
		float* hNeurons = (float*)malloc(NeuronSizeMemory());
		GetNeuronErrorData(hNeurons);

		printf("==Layer Error==\n");
		for (size_t n = 0; n < BatchCount; n++)
		{
			printf("Example %i:\n", n);
			for (size_t d = 0; d < Depth; d++)
			{
				printf("    Filter %i:\n", d);
				for (size_t h = 0; h < Height; h++)
				{
					printf("        ");
					for (size_t w = 0; w < Width; w++)
					{
						float val = hNeurons[(Width * Height * Depth * n) + (Width * Height * d) + (Width * h) + w];
						printf("%f ", val);

					}
					printf("\n");
				}
				if (earlyBreak == 1)
					goto breakForLoops;
			}
			if (earlyBreak == 0)
				goto breakForLoops;
		}
	breakForLoops:
		printf("=========\n");
		free(hNeurons);
	}
	void PrintLayer(int earlyBreak)
	{
		float* hNeurons = (float*)malloc(NeuronSizeMemory());
		GetNeuronData(hNeurons);

		printf("==Layer==\n");
		for (size_t n = 0; n < BatchCount; n++)
		{
			printf("Example %i:\n", n);
			for (size_t d = 0; d < Depth; d++)
			{
				printf("    Filter %i:\n", d);
				for (size_t h = 0; h < Height; h++)
				{
					printf("        ");
					for (size_t w = 0; w < Width; w++)
					{
						float val = hNeurons[(Width * Height * Depth * n) + (Width * Height * d) + (Width * h) + w];
						printf( "%f " , val);

					}
					printf("\n");
				}
				if (earlyBreak == 1)
					goto breakForLoops;
			}
			if (earlyBreak == 0)
				goto breakForLoops;
		}
		breakForLoops:
		printf("=========\n");
		free(hNeurons);
	}
	void PrintWeightsAndBaises()
	{
		PrintWeightsAndBaises(false);
	}
	void PrintWeightsAndBaises(bool earlyBreak)
	{
		if (LayerType != layerType::FeedFowardActivation && LayerType != layerType::FeedFowardSoftmax && LayerType != layerType::Convolutional)
			Error("PrintWeightsAndBaises not supported for this layer type.");
		float* hWeights = (float*)malloc(WeightSizeMemory());
		Try(cudaMemcpy(hWeights, cWeights, WeightSizeMemory(), cudaMemcpyKind::cudaMemcpyDeviceToHost));

		float* hBias = (float*)malloc(BiasSizeMemory());
#ifdef BaisActive
		Try(cudaMemcpy(hBias, cBias, BiasSizeMemory(), cudaMemcpyKind::cudaMemcpyDeviceToHost));
#endif

		printf("==Layer==\n");
		int counter = 0;
		int counterIndepth = 0;
		for (size_t d = 0; d < Depth; d++)
		{
			for (size_t h = 0; h < Height; h++)
			{
				for (size_t w = 0; w < Width; w++)
				{
					printf("Neuron d=%i,h=%i,w=%i. number=%i\n", d,h,w,counter);
#ifdef BaisActive
					printf("bias: %f\n", hBias[counter]);
#endif
					for (size_t z = 0; z < InputDepth; z++)
					{
						for (size_t y = 0; y < InputHeight; y++)
						{
							for (size_t x = 0; x < InputWidth; x++)
							{
								printf("     weight to z=%i,y=%i,x=%i. index=%i  val=%f \n", z,y,x,counterIndepth, hWeights[counterIndepth]);
								counterIndepth++;
							}
						}
						if (earlyBreak && z == 1)
							goto breakAllFors;
					}
					counter++;
					printf("\n");
					if (LayerType == layerType::Convolutional)
						goto breakAllFors;
				}
			}
		}
		breakAllFors:
		printf("=========\n");

		free(hBias);
		free(hWeights);
	}

#if defined(NaNCheck)
	bool FindNaNsInLayerForwards(int LayerIndex)
	{
		float* hNeuron = (float*)malloc(NeuronSizeMemory());
		GetNeuronData(hNeuron);
		bool ret = FindNaNs(hNeuron, NeuronCountMemory(), "hNeuron", LayerIndex);
		free(hNeuron);
		return ret;
	}
	void FindNaNsInLayerBackwards(int LayerIndex)
	{
		float* hNeuronError = (float*)malloc(NeuronSizeMemory());
		GetNeuronErrorData(hNeuronError);
		FindNaNs(hNeuronError, NeuronCountMemory(), "hNeuronError", LayerIndex);

		if (LayerType == layerType::Convolutional || LayerType == layerType::FeedFowardActivation || LayerType == layerType::FeedFowardSoftmax)
		{
			float* hWeights = (float*)malloc(WeightSizeMemory());
			Try(cudaMemcpy(hWeights, cWeights, WeightSizeMemory(), cudaMemcpyKind::cudaMemcpyDeviceToHost));
			FindNaNs(hWeights, WeightCountMemory(), "hWeights", LayerIndex);

			free(hWeights);
		}
		free(hNeuronError);
	}
#endif

	void PrintWeightsAndBaisesError_part1(float* hWeights_temp, float* hBias_temp)
	{
		if (LayerType != layerType::FeedFowardActivation && LayerType != layerType::FeedFowardSoftmax && LayerType != layerType::Convolutional)
			Error("PrintWeightsAndBaises not supported for this layer type.");

		Try(cudaMemcpy(hWeights_temp, cWeights, WeightSizeMemory(), cudaMemcpyKind::cudaMemcpyDeviceToHost));

#ifdef BaisActive
		Try(cudaMemcpy(hBias_temp, cBias, BiasSizeMemory(), cudaMemcpyKind::cudaMemcpyDeviceToHost));
#endif
	}
	void PrintWeightsAndBaisesError_part2(float* hWeights_temp, float* hBias_temp)
	{
		if (LayerType != layerType::FeedFowardActivation && LayerType != layerType::FeedFowardSoftmax && LayerType != layerType::Convolutional)
			Error("PrintWeightsAndBaises not supported for this layer type.");
		float* hWeights = (float*)malloc(WeightSizeMemory());
		Try(cudaMemcpy(hWeights, cWeights, WeightSizeMemory(), cudaMemcpyKind::cudaMemcpyDeviceToHost));

		float* hBias = (float*)malloc(BiasSizeMemory());
#ifdef BaisActive
		Try(cudaMemcpy(hBias, cBias, BiasSizeMemory(), cudaMemcpyKind::cudaMemcpyDeviceToHost));
#endif
		printf("==Layer==\n");
		int counter = 0;
		int counterIndepth = 0;
		printf("Neuron [ALL] number=%i\n", counter);
#ifdef BaisActive
		printf("bias change: %f\n", hBias[counter] - hBias_temp[counter]);
#endif
		for (size_t z = 0; z < InputDepth; z++)
		{
			for (size_t y = 0; y < InputHeight; y++)
			{
				for (size_t x = 0; x < InputWidth; x++)
				{
					printf("     weight change to z=%i,y=%i,x=%i. index=%i  val=%f \n", z, y, x, counterIndepth, hWeights[counterIndepth] - hWeights_temp[counterIndepth]);
					counterIndepth++;
				}
			}

		}
		counter++;
		printf("\n");

		printf("=========\n");
		free(hBias);
		free(hWeights);
		//free(hBias_temp);
		//free(hWeights_temp);
	}
	void Destroy()
	{
		Try(cudaFree(cNeurons));
		Try(cudaFree(cNeuronsErrors));
		if (LayerType == layerType::Example)
			Try(cudaFree(cExamples));
		else
		{
			Try(cudaFree(cWeights));
#ifdef BaisActive
			Try(cudaFree(cBias));
#endif

		}
	}
};

struct Network
{
	float* CommonWorkspace; //for cudnn functions
	size_t CommonWorkspaceSize;

	Layer* Layers;
	Layer* TestingLayers;

	Layer BackupLayer0;
	Layer BackupLayer1;

	size_t LayerCnt;
	size_t BatchCount; //Never zero
	size_t TotalExampleCount; //Never zero
	size_t TotalTestingCount; //Never zero
	size_t TotalTrainingCount; //Never zero

	float* BestPercentCorrect = new float[2]{ -1, -1 };
	float* BestCrossEntropyError = new float[2]{ -1, -1 };
	Network()
	{
		CommonWorkspaceSize = DefaultWorkSpaceSize;
		Try(cudaMalloc(&CommonWorkspace, CommonWorkspaceSize));

		size_t layerCnt = 32; //this number is arb, the array is used as a list

		ThreadSyncMutex.lock();
		FirstWeightIndex = 0;
		if (!FirstWeightsReadyFlag)
			FirstWeightArray = (float**)malloc(sizeof(float) * layerCnt * 2); //2, one for bias and other for weight
		Layer* layers = (Layer*)malloc(layerCnt * sizeof(Layer));
		Layer* TestingLayers = (Layer*)malloc(2 * sizeof(Layer));
		TestingLayers[0] = Layer(); //first
		TestingLayers[1] = Layer(); //final
		for (size_t i = 0; i < layerCnt; i++)
		{
			layers[i] = Layer();
		}

		size_t index = 0;

		layers[index].ConstructExampleLayer(CommonWorkspace, batchCount, TrainingDataCount, ExampleInputDepth, ExampleInputHeight, ExampleInputWidth, ImageData);
		TestingLayers[0].ConstructExampleLayer(CommonWorkspace, batchCount, TestDataCount, ExampleInputDepth, ExampleInputHeight, ExampleInputWidth, &ImageData[layers[index].ExampleCountMemory()]);
		index++;

		layers[index].ConstructConvolutionalLayer(CommonWorkspace, batchCount, layers[index - 1], 64, 5, 5, ActivationFunctionReLu, 1, 1, 0, 0);
		index++;
		layers[index].ConstructPoolingLayer(CommonWorkspace, batchCount, layers[index - 1], 3, 3, cudnnPoolingMode_t::CUDNN_POOLING_MAX, 2, 2, 0, 0);
#ifdef LocalResponseNormalization
		layers[index].LRN = true;
#endif
		index++;

		layers[index].ConstructConvolutionalLayer(CommonWorkspace, batchCount, layers[index - 1], 64 * 1, 5, 5, ActivationFunctionReLu);
		index++;
		layers[index].ConstructPoolingLayer(CommonWorkspace, batchCount, layers[index - 1], 3, 3, cudnnPoolingMode_t::CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING, 2, 2, 0, 0);
#ifdef LocalResponseNormalization
		layers[index].LRN = true;
#endif
		index++;


		layers[index].ConstructFullyConnectedLayer_ActivationFunction(CommonWorkspace, batchCount, layers[index - 1], 256, ActivationFunctionReLu);
		index++;

		/*layers[index].ConstructDropoutLayer(CommonWorkspace, batchCount, layers[index - 1], 0.5f);
		index++;*/

		layers[index].ConstructFullyConnectedLayer_Softmax(CommonWorkspace, batchCount, layers[index - 1], ExampleOutputDepth);
		index++;

		layers[index].ConstructExampleLayer(CommonWorkspace, batchCount, TrainingDataCount, ExampleOutputDepth, layers[index - 1].Height, layers[index - 1].Width, LabelData);
		TestingLayers[1].ConstructExampleLayer(CommonWorkspace, batchCount, TestDataCount, ExampleOutputDepth, layers[index - 1].Height, layers[index - 1].Width, &LabelData[layers[index].ExampleCountMemory()]);
		index++;

		FirstWeightsReadyFlag = true;
		ThreadSyncMutex.unlock();


		if (index > layerCnt)
			Error("index > layerCnt");
		constructor(layers, TestingLayers, index, TrainingDataCount, TestDataCount, batchCount);
		//ValidateRegression();
	}
	void constructor(Layer* layers, Layer* testingLayers, size_t layerCnt, size_t totalTrainingCount, size_t totalTestingCount, size_t batchCount)
	{
		Layers = layers;
		TestingLayers = testingLayers;
		LayerCnt = layerCnt;
		TotalTrainingCount = totalTrainingCount;
		TotalTestingCount = totalTestingCount;
		TotalExampleCount = TotalTrainingCount;
		BatchCount = batchCount;
		BackupLayer0 = Layers[0];
		BackupLayer1 = Layers[LayerCnt - 1];

		if (BatchCount == 0 || TotalExampleCount == 0 || TotalTestingCount == 0 || TotalTrainingCount == 0)
			Error("Invalid arguments.");
		printf("Network ready with (BatchCount=%i, LayerCount=%i, TrainingExampleCount=%i, TestingExampleCount=%i)\n", BatchCount, LayerCnt, TotalTrainingCount, TotalTestingCount);

#ifdef LocalResponseNormalization
		printf("LocalResponseNormalization active with:\n");
		printf("    LocalResponseNormalizationAlpha = %f\n", LocalResponseNormalizationAlpha);
		printf("    LocalResponseNormalizationBeta = %f\n", LocalResponseNormalizationBeta);
		printf("    LocalResponseNormalizationNMin = %i\n", LocalResponseNormalizationNMin);
		printf("    LocalResponseNormalizationNMax = %i\n", LocalResponseNormalizationNMax);
#else
		printf("LocalResponseNormalization inactive.\n");
#endif


#if defined(PauseAndShowActivations)
		Layers[1].PrintWeightsAndBaises(true);
#endif
	}

	void FeedFoward(int startIndex)
	{
		Layers[0].FeedFoward(startIndex);
#if defined(PauseAndShowActivations)
		printf("\n\nActivations for example inputs:");
		Layers[0].PrintLayer(1);
#endif
		for (size_t i = 1; i < LayerCnt - 1; i++)
		{
			Layers[i].FeedFoward(Layers[i - 1]);
#if defined(PauseAndShowActivations)
			printf("Activations for layer %i:", i);
			Layers[i].PrintLayer(1);
#endif
#if defined(NaNCheck)
			if (Layers[i].FindNaNsInLayerForwards(i))
			{
				Layers[i - 1].PrintLayer(1);
				Layers[i].PrintWeightsAndBaises(true);
				Layers[i].PrintLayer(0);
				Layers[i].mod_FeedFoward(Layers[i - 1]);
				Layers[i].PrintLayer(0);

			}
#endif
		}
		Layers[LayerCnt - 1].FeedFoward(startIndex);
#if defined(PauseAndShowActivations)
		printf("Activations for example outputs:");
		Layers[LayerCnt - 1].PrintLayer(0);
#endif
	}

	void BackPropagate(int startIndex)
	{
		Layers[LayerCnt - 1].Backpropagate(Layers[LayerCnt - 2], startIndex);
		for (int i = LayerCnt - 2; i >= 1; i--)
		{
			Layers[i].Backpropagate(Layers[i - 1]);
#if defined(NaNCheck)
			Layers[i].FindNaNsInLayerBackwards(i);
#endif
		}
	}
	void TrainAllBatchesOnce()
	{
		for (size_t i = 0; i < TotalExampleCount; i += BatchCount)
		{
			if (i % TrainStatusUpdate == 0)
			{
				printf("Training %i-%i", i, i + TrainStatusUpdate);
				RewriteLine_MoveCursor();
			}

			FeedFoward(i);
#if defined(PauseAndShowWeightChanges)
			int index = LayerCnt - 2;
			float* hWeights_temp = (float*)malloc(Layers[index].WeightSizeMemory());
			float* hBias_temp = (float*)malloc(Layers[index].BiasSizeMemory());
			Layers[index].PrintWeightsAndBaisesError_part1(hWeights_temp, hBias_temp);
#endif
			BackPropagate(i);
#if defined(PauseAndShowWeightChanges)
			Layers[index].PrintWeightsAndBaisesError_part2(hWeights_temp, hBias_temp);
			free(hBias_temp);
			free(hWeights_temp);

			Pause();
#endif
		}

		printf("All batches trained once with (ExampleCount=%i, LearningRate=%f, MomentumCoefficient=%f, RegressionCoefficient=%f)\n", TotalExampleCount, LearningRate, MomentumCoefficient, RegressionCoefficient);
	}
	void ValidateRegression()
	{
		if (TotalExampleCount < ValidationSetSize * 2)
		{
			Error("Validation set too large.");
		}
		RegressionCoefficient = RegressionCoefficientMax;
		float CoefficientMaxScore = GetValidationScore();

		RegressionCoefficient = RegressionCoefficientMin;
		float CoefficientMinScore = GetValidationScore();

		while (RegressionCoefficientMax - RegressionCoefficientMin > RegressionCoefficientEpsilon)
		{
			RegressionCoefficient = (RegressionCoefficientMax + RegressionCoefficientMin) / 2.0f;
			float newScore = GetValidationScore();
			if (CoefficientMaxScore < CoefficientMinScore)
			{//min is inferior score
				CoefficientMinScore = newScore;
				RegressionCoefficientMin = RegressionCoefficient;
			}
			else
			{//max is inferior score
				CoefficientMaxScore = newScore;
				RegressionCoefficientMax = RegressionCoefficient;
			}
			printf("RegressionCoefficient is (%.20f,%.20f)\n", RegressionCoefficientMin, RegressionCoefficientMax);
		}
		RegressionCoefficient = (RegressionCoefficientMax + RegressionCoefficientMin) / 2.0f;
		printf("RegressionCoefficient = %.20f\n", RegressionCoefficient);
	}
	void RandomiseWeights()
	{
		for (size_t i = 0; i < LayerCnt; i++)
		{
			Layers[i].RandomiseWeights();
		}
	}
	float GetValidationScore()
	{
#define RandomGetCount 1
		float total = 0;
		for (size_t jj = 0; jj < RandomGetCount; jj++)
		{
			RandomiseWeights();
			float oldValidationError = 10e10f;
			while (true)
			{
				bool testMode = false;
				float validationError = 0;
				for (size_t i = 0; i < ValidationSetSize; i += BatchCount)
				{
					FeedFoward(i);
					if (testMode)
					{
						validationError += Layers[LayerCnt - 1].CrossEntropyError(Layers[LayerCnt - 2]);
						RandomiseWeights();
						testMode = false;
					}
					else
					{
						BackPropagate(i);
						testMode = true;
					}
				}
				if (oldValidationError < validationError)
				{
					break;
				}
				oldValidationError = validationError;
			}
			total += oldValidationError;
		}
		return total;
	}

	void SwitchToTestingSet()
	{
		Layers[0] = TestingLayers[0];
		Layers[LayerCnt - 1] = TestingLayers[1];
		TotalExampleCount = TotalTestingCount;
	}
	void SwitchToTrainingSet() //Default settings
	{
		Layers[0] = BackupLayer0;
		Layers[LayerCnt - 1] = BackupLayer1;
		TotalExampleCount = TotalTrainingCount;
	}

	float MeanSquaredError()
	{
		SwitchToTestingSet();
		float sum = 0;
		for (size_t i = 0; i < TotalExampleCount; i += BatchCount)
		{
			FeedFoward(i);
			sum += Layers[LayerCnt - 1].MeanSquaredError(Layers[LayerCnt - 2], i);
		}
		SwitchToTrainingSet();
		return sqrt(sum);
	}
	float CrossEntropyError()
	{
		SwitchToTestingSet();
		float sum = 0;
		for (size_t i = 0; i < TotalExampleCount; i += BatchCount)
		{
			FeedFoward(i);
			sum += Layers[LayerCnt - 1].CrossEntropyError(Layers[LayerCnt - 2]);
		}
		SwitchToTrainingSet();
		return sum;
	}
	float CorrectExamplePercent()
	{
		SwitchToTestingSet();
		size_t sum = 0;
		for (size_t i = 0; i < TotalExampleCount; i += BatchCount)
		{
			FeedFoward(i);
			sum += Layers[LayerCnt - 1].CorrectExampleCount(Layers[LayerCnt - 2]);
		}
		SwitchToTrainingSet();
		return (float)sum / TotalExampleCount;
	}
	void RandomiseDropoutNodes()
	{
		for (size_t i = 0; i < LayerCnt; i++)
		{
			Layers[i].RandomiseDropoutNodes();
		}
		printf("Randomised Dropout Nodes\n");
	}

	void _internalPrintComboError(int index, char* prefex)
	{
		float sum = 0;
		int sumCorrect = 0;
		for (size_t i = 0; i < TotalExampleCount; i += BatchCount)
		{
			FeedFoward(i);
			Layers[LayerCnt - 1].ComboErrorCount(Layers[LayerCnt - 2], &sumCorrect, &sum);
			if (i % 1000 == 0)
			{
				//printf(StrConcat(prefex, " Calculating combo error: %.2f%%"), i / (float)TotalExampleCount * 100);
				//RewriteLine_MoveCursor();
			}
		}
		float percentCorrect = ((float)sumCorrect / TotalExampleCount) * 100;

		ThreadSyncMutex.lock();
		printf(StrConcat(prefex, " CorrectExamplePercent:"));
		if (BestPercentCorrect[index] == -1)
		{
			BestPercentCorrect[index] = percentCorrect;
		}
		else if (BestPercentCorrect[index] < percentCorrect)
		{
			BestPercentCorrect[index] = percentCorrect;
			ConsoleColorGreen();
		}
		else if (BestPercentCorrect[index] > percentCorrect)
		{
			ConsoleColorRed();
		}
		printf(" %.2f%%     ", percentCorrect);
		ConsoleColorReset();
		printf(StrConcat(prefex, " CrossEntropyError:"));
		if (BestCrossEntropyError[index] == -1)
		{
			BestCrossEntropyError[index] = sum;
		}
		else if (BestCrossEntropyError[index] > sum)
		{
			BestCrossEntropyError[index] = sum;
			ConsoleColorGreen();
		}
		else if (BestCrossEntropyError[index] < sum)
		{
			ConsoleColorRed();
		}
		printf(" %f\n", sum);
		ConsoleColorReset();
		ThreadSyncMutex.unlock();
	}
	void PrintComboError()
	{
		SwitchToTestingSet();
		_internalPrintComboError(0, "Testing set: ");
		SwitchToTrainingSet();
		_internalPrintComboError(1, "Training set: ");
	}

	void PrintCrossEntropyError()
	{
		printf("CrossEntropyError: %f\n", CrossEntropyError());
	}
	void PrintMeanError()
	{
		printf("MeanSquaredError: %f\n", MeanSquaredError());
	}
	void PrintCorrectExampleError()
	{
		printf("CorrectExamplePercent: %f%%\n", (CorrectExamplePercent() * 100));
	}
	void PrintOutput()
	{
		for (size_t i = 0; i < TotalExampleCount; i += BatchCount)
		{
			FeedFoward(i);
			Layers[LayerCnt - 2].PrintLayer(false);
		}
	}
	void PrintLayer(int LayerIndex)
	{
		Layers[LayerIndex].PrintLayer(false);
	}
};

int main()
{
	srand(time(NULL));
	ConsoleColorSetup();


#if defined(CreateDummyWeights)
	randomEngine.seed(1234);
#endif

	#pragma region load_cifar10
	ImageData = (float*)malloc((TrainingDataCount + TestDataCount) * (/*image data*/(ExampleInputDepth /*channels*/ * ExampleInputHeight /*height*/ * ExampleInputWidth /*width*/)) * sizeof(float));
	LabelData = (float*)malloc((TrainingDataCount + TestDataCount) * (ExampleOutputDepth) * sizeof(float));

	size_t posImg = 0;
	size_t poslbl = 0;

	byte labelBuffer = 0;
	size_t imageByteCount = 32 * 32 * 3;
	byte* imageBuffer = (byte*)malloc(sizeof(byte) * imageByteCount);

	for (size_t i = 1; i <= 6; i++)
	{
		char * fileName = "C:\\cifar-10\\data_batch_";
		char * num = (char*)malloc(sizeof(char) * 2);
		itoa(i, num, 10);
		fileName = StrConcat(StrConcat(fileName, num), ".bin");
		free(num);
		printf("Loading images from: %s\n", fileName);

		FILE *cfile;
		cfile = fopen(fileName, "rb");  free(fileName);
		for (size_t i = 0; i < 10000; i++)
		{
			int sumTotal = 0;
			fread(&labelBuffer, sizeof(byte), 1, cfile);
			fread(imageBuffer, sizeof(byte), imageByteCount, cfile);
			for (size_t i = 0; i < ExampleOutputDepth; i++)
			{
				LabelData[poslbl + i] = (labelBuffer == i) ? fmaxOutput : fminOutput;
			}
			for (size_t i = 0; i < imageByteCount; i++)
			{
				if (CenterInput)
				{
					ImageData[posImg + i] = imageBuffer[i];
					sumTotal += imageBuffer[i];
				}
				else
				{
					ImageData[posImg + i] = (imageBuffer[i] / 255.0f) * (fmaxInput - fminInput) + fminInput;
				}
			}
			if (CenterInput)
			{
				int average = (sumTotal / imageByteCount);
				for (size_t i = 0; i < imageByteCount; i++)
				{
					ImageData[posImg + i] -= average;
				}
			}
			poslbl += ExampleOutputDepth;
			posImg += imageByteCount;
		}
	}
	free(imageBuffer);

	#pragma endregion

	Try(cudnnCreate(&lib));

	Try(cudnnCreateActivationDescriptor(&ActivationFunctionTanh));
	Try(cudnnSetActivationDescriptor(ActivationFunctionTanh, cudnnActivationMode_t::CUDNN_ACTIVATION_TANH, cudnnNanPropagation_t::CUDNN_PROPAGATE_NAN, 0));

	Try(cudnnCreateActivationDescriptor(&ActivationFunctionReLu));
	Try(cudnnSetActivationDescriptor(ActivationFunctionReLu, cudnnActivationMode_t::CUDNN_ACTIVATION_RELU, cudnnNanPropagation_t::CUDNN_PROPAGATE_NAN, 0));

	printf("BaisActive = ");
#ifdef BaisActive
	printf("true\n");
#else
	printf("false\n");
#endif

	BatchTrainingEpochs = 1000;
	Network net = Network();


	while (true)
	{
		for (size_t k = 0; k < 20; k++)
		{
			net.RandomiseDropoutNodes();
			for (size_t t = 0; t < BatchTrainingEpochs; t++)
				//while (((GetKeyState(VK_SCROLL) & 0x0001) == 0)) //while scrollock off
			{
				net.TrainAllBatchesOnce();
				net.PrintComboError();
			}
		}
		

		printf("\nMerged network.\n");
		DropoutMergedNetwork = true;
		net.PrintComboError();
		DropoutMergedNetwork = false;
		printf("\nUn-merged network.\n\n\n");
	}

	free(LabelData);
	free(ImageData);

	//review - destroy layers and network properly
	Try(cudnnDestroy(lib));

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaError cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }
	OnExit();
    return 0;
}
