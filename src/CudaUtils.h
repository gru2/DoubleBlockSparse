#ifndef UTILS_H
#define UTILS_H

#include <eigen3/Eigen/Core>

namespace CudaUtils
{

void *deviceAllocate(size_t x);
void deviceFree(void *x);

template<typename T>
class Matrix
{
public:
	Matrix();
	Matrix(int rows, int cols, T *data, bool onDevice, bool pinned=false, bool ownsMemory=true);
	Matrix(const Matrix<T> &x);
	Matrix(Matrix<T> &&x);
	~Matrix();
	int rows;
	int cols;
	T *data;
	bool onDevice;
	bool pinned;
	bool ownsMemory;
};

using MatrixF = Matrix<float>;

MatrixF toDevice(const MatrixF &x);
MatrixF toHost(const MatrixF &x);
Eigen::MatrixXf toEigen(const MatrixF &x);
MatrixF toMatrix(const Eigen::MatrixXf &x);

template<typename T>
Matrix<T> allocateMatrixOnDevice(int rows, int cols)
{
	void *data = cuAllocate(sizeof(T) * rows * cols);
	Matrix<T> r(rows, cols, static_cast<T *>(data), true);
	return r;
}

template<typename T>
Matrix<T> allocateMatrixOnHost(int rows, int cols)
{
}

inline MatrixF allocateMatrixOnDeviceF(int rows, int cols)
{
	return allocateMatrixOnDevice<float>(rows, cols);
}

void sgemm(MatrixF &lhs, MatrixF &rhs, MatrixF &r, handle);

class CuBlassHandle
{
public:
	CuBlassHandle();
	~CuBlassHandle();
	void *getHandle();
protected:
	void *handle;
};

};
#endif

