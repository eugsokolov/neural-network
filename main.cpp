// Eugene Sokolov
// Neural Network

#include <iostream>
#include <string>
#include <fstream>
#include "nn.h"

using namespace std;

int main(){

	string resp, f;
	ifstream weightsFile, exampleFile;
	ofstream outFile;
	double lr;
	int epochs;

	cout << "Enter 1 to train a neural network" << endl;
	cout << "Enter 2 to test a neural network" << endl;
	cout << ">>";
	cin >> resp;

	if(resp == "1"){
		cout << "Enter filename for initialization weights: " << endl;
		cout << ">>";
		cin >> resp;
		weightsFile.open(resp.c_str());
	
		cout << "Enter filename for training example: " << endl;
		cout << ">>";
		cin >> resp;
		exampleFile.open(resp.c_str());

		cout << "Enter filename for output: " << endl;
		cout << ">>";
		cin >> resp;
		outFile.open(resp.c_str());

		cout << "Enter learning rate (double): " << endl;
		cout << ">>";
		cin >> lr;
		//lr = stod(resp);
		cout << "Enter number of epochs (int): " << endl;
		cout << ">>";
		cin >> epochs;
		//epochs = stoi(resp.c_str());

		neuralNet *n = new neuralNet(weightsFile);
		n->train(exampleFile, lr, epochs);
		n->print(outFile);
	}

	else if(resp == "2"){
		cout << "Enter filename for trained weights: " << endl;
		cout << ">>";
		cin >> resp;
		weightsFile.open(resp.c_str());
	
		cout << "Enter filename for testing example: " << endl;
		cout << ">>";
		cin >> resp;
		exampleFile.open(resp.c_str());

		cout << "Enter filename for output: " << endl;
		cout << ">>";
		cin >> resp;
		outFile.open(resp.c_str());

		neuralNet *n2 = new neuralNet(weightsFile);
		n2->test(exampleFile, outFile);
		
	}

	cout << "Goodbye" << endl;
}
