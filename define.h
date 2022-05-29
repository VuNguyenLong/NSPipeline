#ifndef __DEFINE_H__
#define __DEFINE_H__

#define SAMPLERATE		16000
#define NCHANELS		1

#define NFFT			512
#define HOP_LENGTH		128
#define SPECTRUM_WIDTH	257
#define NSAMPLES		512
#define NFRAMES			5

#define M_PI			3.1415

#define SERVICE_CONTROL_SWITCH_NOFILTER	0xFF
#define SERVICE_CONTROL_SWITCH_PIPELINE	0xFE

#define MODEL1_PATH		"C:/ProgramData/MiniaudioService/model_1.tflite"
#define MODEL2_PATH		"C:/ProgramData/MiniaudioService/model_2.tflite"
#define MODEL_PATH		"C:/ProgramData/MiniaudioService/model.tflite"

#define IOCTL_PUSH_DATA CTL_CODE(FILE_DEVICE_UNKNOWN, 0x2049, METHOD_BUFFERED, FILE_ANY_ACCESS)

#endif