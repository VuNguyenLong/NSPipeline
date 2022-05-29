#include "pch.h"
#include "fourier.h"
#include "utils.h"
#include "Eigen/Dense"

using namespace std;
using namespace Eigen;

VectorXcf* FourierTransform::forward(VectorXf data)
{
	VectorXf* real = new VectorXf(this->nfft / 2 + 1);
	VectorXf* imag = new VectorXf(this->nfft / 2 + 1);
	memcpy_s(this->in_forward, this->nfft * sizeof(float), data.data(), this->nfft * sizeof(float));

	fftwf_execute(this->forward_plan);
	for (unsigned int i = 0; i <= this->nfft / 2; i++)
	{
		(*real)(i) = this->out_forward[i][0];
		(*imag)(i) = this->out_forward[i][1];
	}

	VectorXcf* result;
	Create_complex_vector(real, imag, result);
	delete real;
	delete imag;

	return result;
}

VectorXf* FourierTransform::inverse(VectorXcf data)
{
	for (unsigned int i = 0; i <= this->nfft / 2; i++)
	{
		this->out_forward[i][0] = data.real()(i) / this->nfft;
		this->out_forward[i][1] = data.imag()(i) / this->nfft;
	}

	fftwf_execute(this->inverse_plan);

	VectorXf* result = new VectorXf(Map<Vector<float, 512>>(this->out_inverse));

	return result;
}