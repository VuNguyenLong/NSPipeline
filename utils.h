#pragma once
#ifndef __UTILS_H__
#define __UTILS_H__

#include "Eigen/Dense"
#include <iostream>
#include <vector>
#include <chrono>
#include <Windows.h>

using namespace std;
using namespace Eigen;

void Create_complex_vector(VectorXf* real, VectorXf* imag, VectorXcf*& c);
void Complex2polar(VectorXcf* c, VectorXf*& mag, VectorXf*& angle);
void Polar2complex(VectorXf* mag, VectorXf* angle, VectorXcf*& c);

void Assert(bool exp, string error);

class Data
{
protected:
	void* data;
	chrono::steady_clock::time_point timestamp;

	Data()
	{
		this->data = nullptr;
		this->timestamp = chrono::steady_clock::now();
	}

	void __init__(void* data)
	{
		this->data = data;
	}

public:
	Data(void* data)
	{
		this->timestamp = chrono::steady_clock::now();
		this->__init__(data);
	}

	void copy(Data item)
	{
		memcpy_s(this, sizeof(Data), &item, sizeof(Data));
	}

	void clone(Data& item)
	{
		memcpy_s(&item, sizeof(Data), this, sizeof(Data));
	}

	void* get_data()
	{
		return this->data;
	}

	chrono::steady_clock::time_point get_timestamp()
	{
		return this->timestamp;
	}
};

class Queue
{
private:
	vector<Data*> queue;
	HANDLE lock;

	chrono::steady_clock::time_point* last_item_time;
public:
	Queue()
	{
		this->lock = CreateMutex(NULL, FALSE, NULL);
		this->last_item_time = nullptr;
	}

	void enqueue(void* data)
	{
		DWORD dwWaitResult;
		Data* in_data = new Data(data);

		dwWaitResult = WaitForSingleObject(this->lock, 5000); // wait this lock for 5s
		if (dwWaitResult == WAIT_OBJECT_0) // lock acquired
		{
			this->queue.push_back(in_data);
		}

		while (!ReleaseMutex(this->lock)) {};
	}

	void* dequeue()
	{
		DWORD dwWaitResult;
		void* result = nullptr;
		dwWaitResult = WaitForSingleObject(this->lock, 5000); // wait this lock for 5s

		if (dwWaitResult == WAIT_OBJECT_0) // lock acquired
		{
			if (this->queue.size() == 0)
			{
				while (!ReleaseMutex(this->lock)) {};
				return nullptr;
			}

			Data* item = this->queue.front();
			if (this->last_item_time == nullptr)
			{
				result = item->get_data();
				this->last_item_time = new chrono::steady_clock::time_point(item->get_timestamp());
			}
			else
			{
				auto elapsed_nanoseconds = chrono::duration_cast<chrono::nanoseconds>(item->get_timestamp() - *this->last_item_time).count();
				result = item->get_data();
				delete this->last_item_time;
				this->last_item_time = new chrono::steady_clock::time_point(item->get_timestamp());

#ifdef __PRINT_QUEUE_RUNTIME__
				cout << elapsed_nanoseconds << endl;
#endif
			}

			this->queue.erase(this->queue.begin());
			delete item;
			while (!ReleaseMutex(this->lock)) {};
		}

		return result;
	}

	size_t size()
	{
		return this->queue.size();
	}
};

#endif