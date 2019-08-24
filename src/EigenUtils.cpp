#include <EigenUtils.h>
#include <CudaUtils.h>

Eigen::MatrixXf EigenUtils::toEigen(const CudaUtils::MatrixF &x)
{
	int rows = x.rows;
	int cols = x.cols;
	Eigen::MatrixXf r(rows, cols);
	float *p = x.data;
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			r(i, j) = p[cols * i + j];
		}
	}
	return r;
}

CudaUtils::MatrixF EigenUtils::toMatrix(const Eigen::MatrixXf &x)
{
	int rows = x.rows();
	int cols = x.cols();
	CudaUtils::MatrixF r = CudaUtils::allocateMatrixOnHostF(rows, cols);
	float *p = r.data;
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			p[cols * i + j] = x(i, j);
		}
	}
	return r;
}
