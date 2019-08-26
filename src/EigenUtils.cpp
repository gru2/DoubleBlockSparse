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

bool EigenUtils::almostEqual(const Eigen::MatrixXf &lhs, const Eigen::MatrixXf &rhs, float tol)
{
	if (lhs.rows() != rhs.rows())
		return false;
	if (lhs.cols() != rhs.cols())
		return false;
	for (int j = 0; j < lhs.rows(); j++)
		for (int i = 0; i < lhs.cols(); i++)
			if (fabs(lhs(j, i) - rhs(j, i)) > tol)
				return false;
	return true;
}
