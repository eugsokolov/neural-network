#ifndef _NEURAL_NET_H
#define _NEURAL_NET_H

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include <iomanip>

class neuralNet{

public:
	class nConn;
	class neuron{
	public:
		double activation;
		double input;
		double error;
		std::vector<nConn> incomingConns;
		std::vector<nConn> outgoingConns;
	};

	class nConn{
	public:
		double weight;
		neuron *connNeuron;
	};

	class training{
	public:
		std::vector<double> inputs;
		std::vector<int> outputs;
	};

	int numLayers;
	std::vector<int> layerSizes;
	std::vector<std::vector<neuron> > layers;	

	neuralNet(std::ifstream &file);
	int train(std::ifstream &f, double lr, int e);
	int test(std::ifstream &f, std::ofstream &g);
//	void evaluateMetrics(std::ofstream &g);
	void evaluateMetrics(std::ofstream &g, std::vector<std::vector<double> > r);
	void print(std::ofstream &file);

	double activationFunc(double input);
	double activationFuncDeriv(double input);

};
#endif
