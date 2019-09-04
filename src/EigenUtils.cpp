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

bool EigenUtils::almostEqual(const Eigen::MatrixXf &lhs, const Eigen::MatrixXf &rhs, float tol, int verbose)
{
	if (lhs.rows() != rhs.rows())
	{
		if (verbose > 0)
			std::cout << "rows does not match.\n"
		return false;
	}
	if (lhs.cols() != rhs.cols())
	{
		if (verbose > 0)
			std::cout << "rows does not match.\n"
		return false;
	}
	float max_err = 0.0f;
	for (int j = 0; j < lhs.rows(); j++)
		for (int i = 0; i < lhs.cols(); i++)
		{
			float t = fabs(lhs(j, i) - rhs(j, i));
			if (t > max_err)
				max_err = t;
		}
	if (max_err > tol)
	{
		if (verbose > 0)
			std::cout << "max err greater then tol. max_err = " << max_err << " tol = " << tol << "\n";
		return false;
	}
	if (verbose > 1)
		std::cout << "max_err = " << max_err << " tol = " << tol << "\n";
	return true;
}
