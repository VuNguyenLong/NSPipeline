#include "pch.h"
#include "utils.h"

void Assert(bool exp, string error)
{
	if (!exp) throw error;
}

void Create_complex_vector(VectorXf* real, VectorXf* imag, VectorXcf*& c)
{
	c = new VectorXcf(real->rows());

	c->real() = *real;
	c->imag() = *imag;
}

void Complex2polar(VectorXcf* c, VectorXf*& mag, VectorXf*& angle)
{
	mag = new VectorXf(c->cwiseAbs());
	angle = new VectorXf(c->cwiseArg());
}

void Polar2complex(VectorXf* mag, VectorXf* angle, VectorXcf*& c)
{
	VectorXf real = mag->cwiseProduct(angle->array().cos().matrix());
	VectorXf imag = mag->cwiseProduct(angle->array().sin().matrix());

	Create_complex_vector(&real, &imag, c);
}