#include "pch.h"
#include "pipeline.h"
#include "fourier.h"
#include "define.h"
#include "utils.h"

void Pipeline::put(float* data)
{
	Assert(data != nullptr, "Data is null");
	this->in->put(data);
}

float* Pipeline::infer()
{
    float* signal = nullptr;

    vector<float>* m, * a;
    this->in->get_input(m, a);

    if (!(m == nullptr) && !(a == nullptr))
    {
        vector<float>* ang = this->in->get_main_frame_angle();

        float* result = this->inference->infer(m);
        this->out->put(result, ang->data());

        signal = new float[HOP_LENGTH];
        memcpy(signal, this->out->get_output().data(), HOP_LENGTH * sizeof(float));

        /*
        for (int i = 0; i < HOP_LENGTH; i++)
            cout << signal[i] << " ";
        cout << endl;
        //*/

        delete[] result;
        delete ang;
        delete m;
        delete a;
    }

    return signal;
}