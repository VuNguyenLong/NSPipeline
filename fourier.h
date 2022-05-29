#ifndef __FOURIER_H__
#define __FOURIER_H__
#include "fftw3.h"
#include "Eigen/Dense"
#include "define.h"
#include <iostream>

using namespace std;
using namespace Eigen;

class FourierTransform
{
	const unsigned int nfft		= NFFT;
	const unsigned int nsamples = NSAMPLES;
	fftwf_plan forward_plan;
	fftwf_plan inverse_plan;

	float* in_forward;
	float* out_inverse;

	fftwf_complex* out_forward; // in_inverse
public:
	FourierTransform()
	{
		this->in_forward	= new float[this->nsamples];
		this->out_inverse	= new float[this->nsamples];
		this->out_forward	= new fftwf_complex[this->nfft];

		fftwf_init_threads();
		fftwf_plan_with_nthreads(1);
		this->forward_plan = fftwf_plan_dft_r2c_1d(this->nfft, this->in_forward, this->out_forward, FFTW_ESTIMATE);
		this->inverse_plan = fftwf_plan_dft_c2r_1d(this->nfft, this->out_forward, this->out_inverse, FFTW_ESTIMATE);
	}

	VectorXcf* forward(VectorXf data);
	VectorXf* inverse(VectorXcf data);

	~FourierTransform()
	{
		fftwf_destroy_plan(this->forward_plan);
		fftwf_destroy_plan(this->inverse_plan);
		delete[] this->in_forward;
		delete[] this->out_forward;
		delete[] this->out_inverse;
		fftwf_cleanup_threads();
		fftwf_cleanup();
	}
};

#endif