#include "nn.h"

using namespace std;

neuralNet::neuralNet(ifstream &file){

//Some initalization
	numLayers = 3;
	layerSizes.resize(numLayers);
	layers.resize(numLayers);
	for(int i = 0; i < numLayers; i++){
	 file >> layerSizes[i];
	 layerSizes[i]++;
	 layers[i].resize(layerSizes[i]);
	}

//Set bias inputs
	for(int i = 0; i<numLayers; i++){
	 layers[i][0].activation = -1;
	 layers[i][0].input = 0;
	 layers[i][0].error = 0;
	}
	

// Set weights
    for(int i=0; i<numLayers-1; i++) {
        for(int j=1; j<layerSizes[i+1]; j++) {
            for (int k=0; k<layerSizes[i]; k++) {
		double weight;
		file >> weight;
		nConn incomingConnection;
		incomingConnection.weight = weight;
		incomingConnection.connNeuron = &layers[i][k];
		layers[i+1][j].incomingConns.push_back(incomingConnection);

		nConn outgoingConnection;
		outgoingConnection.weight = weight;
		outgoingConnection.connNeuron = &layers[i+1][j];
		layers[i][k].outgoingConns.push_back(outgoingConnection);
	   }
	}
   }
}

int neuralNet::train(ifstream &trainDataFile, double learnRate, int epoch){

//cout << "in train function.." << endl;
	vector<training> examples;
	int numExamples=0, numInputs=0, numOutputs=0;
	trainDataFile >> numExamples >> numInputs >> numOutputs;
//Read into memory
//cout << "reading into mem" <<endl;
	examples.resize(numExamples);
	for(int i = 0; i< numExamples; i++){
		examples[i].inputs.resize(numInputs);
		for(int j=0; j<numInputs; j++)
			trainDataFile >> examples[i].inputs[j];

		examples[i].outputs.resize(numOutputs);
		for(int j=0; j<numOutputs; j++)
			trainDataFile >> examples[i].outputs[j];
	}
	
//cout << "training.." << endl;
//Train
	int outLayerI = numLayers - 1;
	for(int e = 0; e < epoch; e++){
	 for(int i = 0; i<numExamples; i++){

	  for(int nodeI=0; nodeI<numInputs; nodeI++)
	   layers[0][nodeI+1].activation=examples[i].inputs[nodeI];

	//Prop input forward
	  for(int layerI=1; layerI<numLayers; layerI++){
		for(int nodeI=1; nodeI<layerSizes[layerI];nodeI++){
		 layers[layerI][nodeI].input = 0;
		 vector<nConn>::iterator it;
	 	 it=layers[layerI][nodeI].incomingConns.begin();
		 while(it!=layers[layerI][nodeI].incomingConns.end()){
			layers[layerI][nodeI].input += it->weight * it->connNeuron->activation;
			it++;
		 }
		 layers[layerI][nodeI].activation=activationFunc(layers[layerI][nodeI].input);

		}
	  }

	//Prop errors back
	  for(int nodeI=1; nodeI<layerSizes[outLayerI]; nodeI++)
	  layers[outLayerI][nodeI].error = activationFuncDeriv(layers[outLayerI][nodeI].input) * (examples[i].outputs[nodeI-1] - layers[outLayerI][nodeI].activation);

	  for(int layerI=outLayerI-1; layerI>0;layerI--){
		for(int nodeI=1; nodeI<layerSizes[layerI]; nodeI++){
		 double sum = 0;
		 vector<nConn>::iterator it;
		 it = layers[layerI][nodeI].outgoingConns.begin();
		 while(it != layers[layerI][nodeI].outgoingConns.end()){
		  sum += it->weight * it->connNeuron->error;
		  it++;
		 }
		 layers[layerI][nodeI].error = activationFuncDeriv(layers[layerI][nodeI].input)*sum;
		}
	  }

	  for(int layerI=1; layerI<numLayers; layerI++){
		for(int nodeI=1; nodeI<layerSizes[layerI];nodeI++){
		 vector<nConn>::iterator it;
		 it = layers[layerI][nodeI].incomingConns.begin();
		 while(it != layers[layerI][nodeI].incomingConns.end()){
		  it->weight = it->weight + learnRate * it->connNeuron->activation*layers[layerI][nodeI].error;
		  it->connNeuron->outgoingConns[nodeI-1].weight = it->weight;
		  it++;
		 }
		}
	  }

	 }

	}

	return 0;
}

int neuralNet::test(ifstream &testDataFile, ofstream &outputFile){
//cout << "in test function.." << endl;

	vector<training> examples;
	vector<vector<double> > results;
	int numExamples=0, numInputs=0, numOutputs=0;

//cout << "read test ex into mem" << endl;
//Read examples into memory

	testDataFile >> numExamples >> numInputs >> numOutputs;
	examples.resize(numExamples);
	results.resize(numOutputs);
	for(int i = 0; i<numExamples; i++){
		examples[i].inputs.resize(numInputs);
		examples[i].outputs.resize(numOutputs);
		for(int j=0;j<numInputs;j++)
		 testDataFile >> examples[i].inputs[j]; 

		for(int j=0;j<numOutputs;j++){
		 testDataFile >> examples[i].outputs[j]; 
		 if(i==0){
			results[j].resize(4);
			for(int k=0; k<4;k++)
			  results[j][k] = 0;
		 }
		}
	}


//cout << "testing.." << endl;
//Test
	int outLayerI = numLayers - 1;

	for(int i = 0; i < numExamples; i++){
	 for(int j = 0; j < numInputs;j++)
	 layers[0][j+1].activation=examples[i].inputs[j];

	//Prop input forward	
	 for(int layerI=1; layerI < numLayers; layerI++){
		for(int nodeI=1; nodeI<layerSizes[layerI];nodeI++){
		 layers[layerI][nodeI].input = 0;
		 vector<nConn>::iterator it;
		 it=layers[layerI][nodeI].incomingConns.begin();
		 while(it != layers[layerI][nodeI].incomingConns.end()){
		 	layers[layerI][nodeI].input += it->weight*it->connNeuron->activation;
			it++;
		 }
		 layers[layerI][nodeI].activation = activationFunc(layers[layerI][nodeI].input);
		}
	 }
	//Threshold output
	 for(int nodeI=1; nodeI<layerSizes[outLayerI];nodeI++){
		if(layers[outLayerI][nodeI].activation >= 0.5){
		 if(examples[i].outputs[nodeI-1])
			results[nodeI-1][0]++;
		 else
		 	results[nodeI-1][1]++;
		}
		else{
		 if(examples[i].outputs[nodeI-1])
		 	results[nodeI-1][2]++;
		 else
		 	results[nodeI-1][3]++;
		}
	 }
	
	}

//cout << "evaluating.."<<endl;
//Evaluate the test
	outputFile << "Evaluation Metrics: " << endl;
	outputFile << std::setprecision(3) << std::fixed;
	double A = 0, B = 0, C = 0, D = 0;
	double avg_accuracy = 0, avg_precision = 0, avg_recall = 0, avg_f1 = 0;
	double oa_accuracy = 0, oa_precision = 0, oa_recall = 0, oa_f1 = 0;

	for(int i=0; i<numOutputs; i++){

		A += results[i][0];	
		B += results[i][1];	
		C += results[i][2];	
		D += results[i][3];	
	
		outputFile << (int)results[i][0] << " ";	
		outputFile << (int)results[i][1] << " ";	
		outputFile << (int)results[i][2] << " ";	
		outputFile << (int)results[i][3] << " ";	

		oa_accuracy = (results[i][0]+results[i][3]) / (results[i][0]+results[i][1]+results[i][2]+results[i][3]);
		oa_precision = results[i][0]/(results[i][0]+results[i][1]);
		oa_recall = results[i][0]/(results[i][0] + results[i][2]);
		oa_f1 = (2*oa_precision*oa_recall)/(oa_precision + oa_recall);

		outputFile << oa_accuracy << " " << oa_precision << " " << oa_recall << " " << oa_f1 << endl;

		avg_accuracy += oa_accuracy;
		avg_precision += oa_precision;
		avg_recall += oa_recall;
	
	}

//MicroAveraging
	oa_accuracy = (A + D) / (A + B + C + D);
	oa_precision = A / (A + B);
	oa_recall = A / (A + C);
	oa_f1 = (2 * oa_precision * oa_recall) / (oa_precision + oa_recall);
	outputFile << "Overall scores: " << endl;
	outputFile << oa_accuracy << " " << oa_precision << " " << oa_recall << " " << oa_f1 << endl;

//MacroAveraging
	avg_accuracy /= numOutputs;
	avg_precision /= numOutputs;
	avg_recall /= numOutputs;
	avg_f1 = (2 * avg_precision * avg_recall) / (avg_precision + avg_recall);
	outputFile << "Average scores: " << endl;
	outputFile << avg_accuracy << " " << avg_precision << " " << avg_recall << " " << avg_f1 << endl;

}

void neuralNet::print(ofstream &file){

	file << std::setprecision(3) << std::fixed;
	int outLayerI = numLayers - 1;
	for(int i = 0; i<numLayers; i++){
		if(i != 0)
		file << " ";
	    file << layerSizes[i]-1;	
	}
	file << endl;
	for(int layerI=1; layerI < numLayers; layerI++){
	 for(int nodeI=1; nodeI<layerSizes[layerI]; nodeI++){
	  vector<nConn>::iterator it;
	  it = layers[layerI][nodeI].incomingConns.begin();
	  while(it != layers[layerI][nodeI].incomingConns.end()){
		if(it!=layers[layerI][nodeI].incomingConns.begin())
		 file << " ";
	   	file<<it->weight;
	   	it++;
	  }
	 file<<endl;
	 }
	}

}

double neuralNet::activationFunc(double input){
	return 1.0/(1.0 + exp(-input));
}

double neuralNet::activationFuncDeriv(double input){
	return activationFunc(input)*(1.0-activationFunc(input));
}
