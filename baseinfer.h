#ifndef __BASEINFER_H__
#define __BASEINFER_H__

#include <vector>
using namespace std;

class BaseInput
{
public:
	virtual void put(float* data) = 0;
	virtual void get_input(vector<float>*& mag, vector<float>*& angle) = 0;
	virtual vector<float>* get_main_frame_mag() = 0;
	virtual vector<float>* get_main_frame_angle() = 0;
};


class BaseOutput
{
public:
	virtual void put(float* mag, float* angle) = 0;
	virtual vector<float> get_output() = 0;
};

class BaseInference
{
public:
	virtual float* infer(vector<float>* data) = 0;
	virtual void reset() = 0;
};

#endif