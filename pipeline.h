#ifndef __PIPELINE_H__
#define __PIPELINE_H__

#include "baseinfer.h"

class Pipeline
{
	BaseInput* in;
	BaseOutput* out;
	BaseInference* inference;
public:
	Pipeline(BaseInput* in, BaseOutput* out, BaseInference* infer)
	{
		this->in		= in;
		this->out		= out;
		this->inference	= infer;
	}


	void put(float* data);
	float* infer();
};

#endif