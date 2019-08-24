#ifndef EIGEN_UTILS_H
#define EIGEN_UTILS_H

#include <Eigen/Core>
#include <CudaUtils.h>

namespace EigenUtils
{

Eigen::MatrixXf toEigen(const CudaUtils::MatrixF &x);
CudaUtils::MatrixF toMatrix(const Eigen::MatrixXf &x);

};
#endif
