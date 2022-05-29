#ifndef __INFERENCE_H__
#define __INFERENCE_H__

#include "define.h"
#include "Eigen/Dense"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/c/c_api.h"
#include "fourier.h"
#include "baseinfer.h"
#include "utils.h"
#include <vector>

using namespace std;
using namespace Eigen;

class Input : public BaseInput
{
	const int main_frame_idx = SPECTRUM_WIDTH;
	vector<float> mag_queue;
	vector<float> angle_queue;
	FourierTransform* fourier;

	VectorXf* window;
	float norm_coff;
public:
	Input(FourierTransform& fourier)
	{
		this->fourier = &fourier;
		this->window = new VectorXf(NSAMPLES);

		for (int i = 0; i < NSAMPLES; i++)
			(*this->window)(i) = 0.5f - 0.5f * cosf((2.f * M_PI * i) / (NSAMPLES + 0.f));
		this->norm_coff = 1.f;
	}

	void put(float*);
	void get_input(vector<float>*&, vector<float>*&);
	vector<float>* get_main_frame_mag();
	vector<float>* get_main_frame_angle();

	~Input()
	{
		delete this->window;
	}
};

class Output : public BaseOutput
{
	vector<float> buffer; // 512 samples computed from prev computation
	FourierTransform* fourier;
	VectorXf* window;
public:
	Output(FourierTransform& fourier)
	{
		this->fourier = &fourier;
		for (int i = 0; i < 512; i++)
			this->buffer.push_back(0.f);

		this->window = new VectorXf(NSAMPLES);
		for (int i = 0; i < NSAMPLES; i++)
			(*this->window)(i) = 0.5f - 0.5f * cosf((2.f * M_PI * i) / (NSAMPLES + 0.f));
	}

	void put(float*, float*);
	vector<float> get_output();
};


class Inference_2Models : public BaseInference
{
	TfLiteModel* model_1, * model_2;
	TfLiteInterpreter* interpreter_model_1, * interpreter_model_2;

	TfLiteInterpreterOptions* options;

	TfLiteTensor* mag_1_in, * h_1_in, * c_1_in, * mag_2_in, * out_1_in, * h_2_in, * c_2_in;
	const TfLiteTensor* mag_1_out, * h_1_out, * c_1_out, * mag_2_out, * h_2_out, * c_2_out;


	float* h_1_cache, * c_1_cache, * h_2_cache, * c_2_cache;
public:
	Inference_2Models(string model_1_path, string model_2_path);
	float* infer(vector<float>* mag);
	void reset()
	{
		this->reset_state();
	}

	void reset_state();
	~Inference_2Models();
};



class Inference_Combined : public BaseInference
{
	TfLiteModel* model;
	TfLiteInterpreter* interpreter_model;
	TfLiteInterpreterOptions* options;

	TfLiteTensor* h_1_in, * h_2_in, * c_1_in, * c_2_in, * mag;
	const TfLiteTensor* h_1_out, * h_2_out, * c_1_out, * c_2_out, * mask;

	float* h_1_cache, * c_1_cache, * h_2_cache, * c_2_cache;
public:
	Inference_Combined(string model_path);
	float* infer(vector<float>* mag);
	void reset()
	{
		this->reset_state();
	}

	void reset_state();
	~Inference_Combined();
};

#endif
