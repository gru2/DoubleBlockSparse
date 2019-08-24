#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <memory>
#include <stdlib.h>

namespace CudaUtils
{

void *deviceAllocate(size_t x);
void deviceFree(void *x);
void memcpyDevice(void *src, void *dst, int len);
void memcpyDeviceToHost(void *src, void *dst, int len);
void memcpyHostToDevice(void *src, void *dst, int len);

template<typename T>
class Matrix
{
public:
	Matrix();
	Matrix(int rows, int cols, T *data, bool onDevice, bool pinned=false, bool ownsMemory=true);
	Matrix(const Matrix<T> &x);
	Matrix(Matrix<T> &&x);
	~Matrix();
	void freeData();
	int rows;
	int cols;
	T *data;
	bool onDevice;
	bool pinned;
	bool ownsMemory;
};

template<typename T>
Matrix<T> allocateMatrixOnDevice(int rows, int cols)
{
	void *data = deviceAllocate(sizeof(T) * rows * cols);
	Matrix<T> r(rows, cols, static_cast<T *>(data), true);
	return r;
}

template<typename T>
Matrix<T> allocateMatrixOnHost(int rows, int cols)
{
	void *data = malloc(sizeof(T) * rows * cols);
	Matrix<T> r(rows, cols, static_cast<T *>(data), false);
	return r;
}

using MatrixF = Matrix<float>;

inline MatrixF allocateMatrixOnDeviceF(int rows, int cols)
{
	return allocateMatrixOnDevice<float>(rows, cols);
}

inline MatrixF allocateMatrixOnHostF(int rows, int cols)
{
	return allocateMatrixOnHost<float>(rows, cols);
}

template<typename T>
Matrix<T> toDevice(const Matrix<T> &x)
{
	Matrix<T> r = allocateMatrixOnDevice<T>(x.rows, x.cols);
	memcpyHostToDevice(x.getData(), r.getData(), x.rows * x.cols * sizeof(T));
	return r;
}

template<typename T>
void toDevice(Matrix<T> &dst, Matrix<T> &src)
{
	memcpyHostToDevice(src.getData(), dst.getData(), dst.rows * dst.cols * sizeof(T));
}

template<typename T>
Matrix<T> toHost(const Matrix<T> &x)
{
	Matrix<T> r = allocateMatrixOnDevice<T>(x.rows, x.cols);
	memcpyDeviceToHost(x.getData(), r.getData(), x.rows * x.cols * sizeof(T));
	return r;
}

template<typename T>
void toHost(Matrix<T> &dst, const Matrix<T> &src)
{
	memcpyDeviceToHost(src.getData(), dst.getData(), dst.rows * dst.cols * sizeof(T));
}

template<typename T>
void copyDataOnHost(Matrix<T> &dst, const Matrix<T> &src)
{
	memcpy(src.getData(), dst.getData(), dst.rows * dst.cols * sizeof(T));
}

template<typename T>
void copyDataOnDevice(Matrix<T> &dst, const Matrix<T> &src)
{
	memcpyDevice(src.getData(), dst.getData(), dst.rows * dst.cols * sizeof(T));
}

template<typename T>
Matrix<T>::Matrix() : rows(0), cols(0), data(0), onDevice(false), pinned(false), ownsMemory(true)
{ }

template<typename T>
Matrix<T>::Matrix(int rows_, int cols_, T *data_, bool onDevice_, bool pinned_, bool ownsMemory_) :
	rows(rows_), cols(cols_), data(data_), onDevice(onDevice_), pinned(pinned_), ownsMemory(ownsMemory_)
{ }

template<typename T>
Matrix<T>::Matrix(const Matrix<T> &x)
{
	if (!x.ownsMemory)
	{
		freeData();
		rows = x.rows;
		cols = x.cols;
		data = x.data;
		onDevice = x.onDevice;
		pinned = x.pinned;
		ownsMemory = x.ownsMemory;
		return;
	}

	if (rows == x.rows && cols == x.cols && onDevice == x.onDevice && pinned == x.pinned)
	{
		if (onDevice)
			copyDataOnDeice(*this, x);
		else
			copyDataOnHost(*this, x);
		return;
	}

	freeData();
	rows = x.rows;
	cols = x.cols;
	onDevice = x.onDevice;
	pinned = x.pinned;
	ownsMemory = x.ownsMemory;
	if (onDevice)
	{
		data = deviceAllocate(sizeof(T) * rows * cols);
		copyDataOnDeice(*this, x);
	}
	else
	{
		data = malloc(sizeof(T) * rows * cols);
		copyDataOnHost(*this, x);
	}
}

template<typename T>
Matrix<T>::~Matrix()
{
	freeData();
}

template<typename T>
void Matrix<T>::freeData()
{
	if (ownsMemory && data)
	{
		if (onDevice)
			deviceFree(data);
		else
			free(data);
	}

	data = 0;
	cols = 0;
	rows = 0;
	onDevice = false;
	pinned = false;
	ownsMemory = true;
}

class CuBlasHandle
{
public:
	CuBlasHandle();
	~CuBlasHandle();
	void *getHandle();
protected:
	void *handle;
};

void gemm(MatrixF &lhs, MatrixF &rhs, MatrixF &r, CuBlasHandle &handle);

};
#endif
