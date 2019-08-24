#include <iostream>
#include <CudaUtils.h>
#include <Usutf.h>
#include <CudaUtils.h>
#include <EigenUtils.h>
#include <Eigen/Core>

USUTF_TEST(testDeviceAllocate)
{
	void *x = CudaUtils::deviceAllocate(1);
	Usutf::test(x != 0);
	CudaUtils::deviceFree(x);
}

USUTF_TEST(testEigenUtils_toEigen_and_toMatrix)
{
	Eigen::MatrixXf a_ref(2, 3);
	a_ref <<
	1.0f, 2.3f, -1.3f,
	4.5f, 2.2f, 11.4f;
	CudaUtils::MatrixF a = EigenUtils::toMatrix(a_ref);
	Eigen::MatrixXf a_test = EigenUtils::toEigen(a);
	Usutf::test(a_test == a_ref);
}

USUTF_TEST(testCudaUtils_toDevice_and_toHost)
{
	Eigen::MatrixXf a_ref(2, 3);
	a_ref <<
	1.0f, 2.3f, -1.3f,
	4.5f, 2.2f, 11.4f;
	CudaUtils::MatrixF a_device = CudaUtils::toDevice(EigenUtils::toMatrix(a_ref));
	CudaUtils::MatrixF a_host = CudaUtils::toHost(a_device);
	Eigen::MatrixXf a_test = EigenUtils::toEigen(a_host);
	Usutf::test(a_test == a_ref);
}

int main(int argc, char *argv[])
{
	std::cout << "test lib double_block_sparse...\n";	
	int r = Usutf::runTests(argc, argv);
	return r;
}
