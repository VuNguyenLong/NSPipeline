#include "pch.h"
#include "infer.h"

void Input::put(float* data)
{
	VectorXf* frame = new VectorXf(Map<Vector<float, NSAMPLES>>(data));
	VectorXcf* spectrum = this->fourier->forward(frame->cwiseProduct((*(this->window))));
	VectorXf* m, * a;

	Complex2polar(spectrum, m, a);
	this->mag_queue.insert(
		this->mag_queue.end(),
		m->data(),
		m->data() + SPECTRUM_WIDTH
	);
	this->angle_queue.insert(
		this->angle_queue.end(),
		a->data(),
		a->data() + SPECTRUM_WIDTH
	);

	if (this->mag_queue.size() > NFRAMES * SPECTRUM_WIDTH)
		this->mag_queue.erase(
			this->mag_queue.begin(),
			this->mag_queue.begin() + SPECTRUM_WIDTH
		);

	if (this->angle_queue.size() > NFRAMES * SPECTRUM_WIDTH)
		this->angle_queue.erase(
			this->angle_queue.begin(),
			this->angle_queue.begin() + SPECTRUM_WIDTH
		);

	delete frame;
	delete spectrum;
	delete m;
	delete a;
}

void Input::get_input(vector<float>*& mag, vector<float>*& angle)
{
	if (this->mag_queue.size() < NFRAMES * SPECTRUM_WIDTH)
	{
		mag = nullptr;
		angle = nullptr;
	}
	else
	{
		mag = new vector<float>(this->mag_queue);
		angle = new vector<float>(this->angle_queue);
	}
}

vector<float>* Input::get_main_frame_mag()
{
	if (this->mag_queue.size() >= NFRAMES * SPECTRUM_WIDTH)
	{
		vector<float>* _m = new vector<float>();
		_m->insert(
			_m->end(),
			this->mag_queue.data() + this->main_frame_idx,
			this->mag_queue.data() + this->main_frame_idx + SPECTRUM_WIDTH
		);
		return _m;
	}

	return nullptr;
}

vector<float>* Input::get_main_frame_angle()
{
	if (this->angle_queue.size() >= NFRAMES * SPECTRUM_WIDTH)
	{
		vector<float>* _a = new vector<float>();
		_a->insert(
			_a->end(),
			this->angle_queue.data() + this->main_frame_idx,
			this->angle_queue.data() + this->main_frame_idx + SPECTRUM_WIDTH
		);
		return _a;
	}

	return nullptr;
}

void Output::put(float* mag, float* angle)
{
	VectorXf* m = new VectorXf(Map<Vector<float, SPECTRUM_WIDTH>>(mag));
	VectorXf* a = new VectorXf(Map<Vector<float, SPECTRUM_WIDTH>>(angle));

	VectorXcf* c;
	Polar2complex(m, a, c);

	VectorXf* signal;
	signal = this->fourier->inverse(*c);

	this->buffer.erase(this->buffer.begin(), this->buffer.begin() + HOP_LENGTH);
	for (int i = 0; i < NSAMPLES; i++)
	{
		if (i < NSAMPLES - HOP_LENGTH)
		{
			this->buffer[i] += (*signal)(i) * (*this->window)(i);
			if (this->buffer[i] > 1) this->buffer[i] = 1;
			else if (this->buffer[i] < -1) this->buffer[i] = -1;
		}
		else this->buffer.push_back((*signal)(i) * (*this->window)(i));
	}

	delete m;
	delete a;
	delete c;
	delete signal;
}

vector<float> Output::get_output()
{
	vector<float> re;
	re.insert(re.end(), this->buffer.data(), this->buffer.data() + HOP_LENGTH);
	return re;
}


Inference_2Models::Inference_2Models(string model_1_path, string model_2_path)
{
	this->model_1 = TfLiteModelCreateFromFile(model_1_path.c_str());
	this->model_2 = TfLiteModelCreateFromFile(model_2_path.c_str());
	this->options = TfLiteInterpreterOptionsCreate();
	TfLiteInterpreterOptionsSetNumThreads(options, 2);

	this->interpreter_model_1 = TfLiteInterpreterCreate(this->model_1, options);
	this->interpreter_model_2 = TfLiteInterpreterCreate(this->model_2, options);

	Assert(TfLiteInterpreterAllocateTensors(this->interpreter_model_1) == TfLiteStatus::kTfLiteOk, "Error");
	Assert(TfLiteInterpreterAllocateTensors(this->interpreter_model_2) == TfLiteStatus::kTfLiteOk, "Error");

	this->c_1_in = TfLiteInterpreterGetInputTensor(this->interpreter_model_1, 0);
	this->h_1_in = TfLiteInterpreterGetInputTensor(this->interpreter_model_1, 1);
	this->mag_1_in = TfLiteInterpreterGetInputTensor(this->interpreter_model_1, 2);

	this->c_2_in = TfLiteInterpreterGetInputTensor(this->interpreter_model_2, 0);
	this->out_1_in = TfLiteInterpreterGetInputTensor(this->interpreter_model_2, 1);
	this->h_2_in = TfLiteInterpreterGetInputTensor(this->interpreter_model_2, 2);
	this->mag_2_in = TfLiteInterpreterGetInputTensor(this->interpreter_model_2, 3);

	this->c_1_out = TfLiteInterpreterGetOutputTensor(this->interpreter_model_1, 0);
	this->mag_1_out = TfLiteInterpreterGetOutputTensor(this->interpreter_model_1, 2);
	this->h_1_out = TfLiteInterpreterGetOutputTensor(this->interpreter_model_1, 1);

	this->h_2_out = TfLiteInterpreterGetOutputTensor(this->interpreter_model_2, 1);
	this->mag_2_out = TfLiteInterpreterGetOutputTensor(this->interpreter_model_2, 0);
	this->c_2_out = TfLiteInterpreterGetOutputTensor(this->interpreter_model_2, 2);

	this->h_1_cache = new float[64];
	this->h_2_cache = new float[64];
	this->c_1_cache = new float[64];
	this->c_2_cache = new float[64];

	this->reset_state();
}


Inference_2Models::~Inference_2Models()
{
	delete[] this->h_1_cache;
	delete[] this->h_2_cache;
	delete[] this->c_1_cache;
	delete[] this->c_2_cache;

	TfLiteInterpreterDelete(this->interpreter_model_1);
	TfLiteInterpreterDelete(this->interpreter_model_2);

	TfLiteInterpreterOptionsDelete(this->options);

	TfLiteModelDelete(this->model_1);
	TfLiteModelDelete(this->model_2);
}


float* Inference_2Models::infer(vector<float>* mag)
{
	TfLiteTensorCopyFromBuffer(this->mag_1_in, mag->data(), mag->size() * sizeof(float));
	TfLiteTensorCopyFromBuffer(this->h_1_in, this->h_1_cache, 64 * sizeof(float));
	TfLiteTensorCopyFromBuffer(this->c_1_in, this->c_1_cache, 64 * sizeof(float));

	TfLiteInterpreterInvoke(this->interpreter_model_1);

	float* out_1_in = new float[128];
	TfLiteTensorCopyToBuffer(this->mag_1_out, out_1_in, 128 * sizeof(float));
	TfLiteTensorCopyToBuffer(this->h_1_out, this->h_1_cache, 64 * sizeof(float));
	TfLiteTensorCopyToBuffer(this->c_1_out, this->c_1_cache, 64 * sizeof(float));


	TfLiteTensorCopyFromBuffer(this->mag_2_in, mag->data(), mag->size() * sizeof(float));
	TfLiteTensorCopyFromBuffer(this->out_1_in, out_1_in, 128 * sizeof(float));
	TfLiteTensorCopyFromBuffer(this->h_2_in, this->h_2_cache, 64 * sizeof(float));
	TfLiteTensorCopyFromBuffer(this->c_2_in, this->c_2_cache, 64 * sizeof(float));

	TfLiteInterpreterInvoke(this->interpreter_model_2);

	TfLiteTensorCopyToBuffer(this->h_2_out, this->h_2_cache, 64 * sizeof(float));
	TfLiteTensorCopyToBuffer(this->c_2_out, this->c_2_cache, 64 * sizeof(float));

	float* result = new float[SPECTRUM_WIDTH];
	TfLiteTensorCopyToBuffer(this->mag_2_out, result, SPECTRUM_WIDTH * sizeof(float));

	delete[] out_1_in;
	return result;
}

void Inference_2Models::reset_state()
{
	for (int i = 0; i < 64; i++)
	{
		this->h_1_cache[i] = 0;
		this->c_1_cache[i] = 0;

		this->h_2_cache[i] = 0;
		this->c_2_cache[i] = 0;
	}
}





Inference_Combined::Inference_Combined(string model_path)
{
	this->model = TfLiteModelCreateFromFile(model_path.c_str());
	this->options = TfLiteInterpreterOptionsCreate();
	TfLiteInterpreterOptionsSetNumThreads(options, 2);

	this->interpreter_model = TfLiteInterpreterCreate(this->model, options);
	Assert(TfLiteInterpreterAllocateTensors(this->interpreter_model) == TfLiteStatus::kTfLiteOk, "Error");

	this->c_1_in = TfLiteInterpreterGetInputTensor(this->interpreter_model, 0);
	this->c_2_in = TfLiteInterpreterGetInputTensor(this->interpreter_model, 1);
	this->h_1_in = TfLiteInterpreterGetInputTensor(this->interpreter_model, 2);
	this->h_2_in = TfLiteInterpreterGetInputTensor(this->interpreter_model, 3);
	this->mag = TfLiteInterpreterGetInputTensor(this->interpreter_model, 4);

	this->mask = TfLiteInterpreterGetOutputTensor(this->interpreter_model, 1);
	this->c_1_out = TfLiteInterpreterGetOutputTensor(this->interpreter_model, 0);
	this->h_1_out = TfLiteInterpreterGetOutputTensor(this->interpreter_model, 4);
	this->c_2_out = TfLiteInterpreterGetOutputTensor(this->interpreter_model, 3);
	this->h_2_out = TfLiteInterpreterGetOutputTensor(this->interpreter_model, 2);

	this->h_1_cache = new float[64];
	this->h_2_cache = new float[64];
	this->c_1_cache = new float[64];
	this->c_2_cache = new float[64];

	this->reset_state();
}

float* Inference_Combined::infer(vector<float>* mag)
{
	TfLiteTensorCopyFromBuffer(this->mag, mag->data(), mag->size() * sizeof(float));
	TfLiteTensorCopyFromBuffer(this->h_1_in, this->h_1_cache, 64 * sizeof(float));
	TfLiteTensorCopyFromBuffer(this->c_1_in, this->c_1_cache, 64 * sizeof(float));
	TfLiteTensorCopyFromBuffer(this->h_2_in, this->h_2_cache, 64 * sizeof(float));
	TfLiteTensorCopyFromBuffer(this->c_2_in, this->c_2_cache, 64 * sizeof(float));

	TfLiteInterpreterInvoke(this->interpreter_model);

	TfLiteTensorCopyToBuffer(this->h_1_out, this->h_1_cache, 64 * sizeof(float));
	TfLiteTensorCopyToBuffer(this->c_1_out, this->c_1_cache, 64 * sizeof(float));
	TfLiteTensorCopyToBuffer(this->h_2_out, this->h_2_cache, 64 * sizeof(float));
	TfLiteTensorCopyToBuffer(this->c_2_out, this->c_2_cache, 64 * sizeof(float));

	float* result = new float[SPECTRUM_WIDTH];
	TfLiteTensorCopyToBuffer(this->mask, result, SPECTRUM_WIDTH * sizeof(float));
	return result;
}

void Inference_Combined::reset_state()
{
	for (int i = 0; i < 64; i++)
	{
		this->h_1_cache[i] = 0;
		this->c_1_cache[i] = 0;

		this->h_2_cache[i] = 0;
		this->c_2_cache[i] = 0;
	}
}


Inference_Combined::~Inference_Combined()
{
	delete[] this->h_1_cache;
	delete[] this->h_2_cache;
	delete[] this->c_1_cache;
	delete[] this->c_2_cache;

	TfLiteInterpreterDelete(this->interpreter_model);
	TfLiteInterpreterOptionsDelete(this->options);
	TfLiteModelDelete(this->model);
}