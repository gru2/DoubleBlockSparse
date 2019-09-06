#include <iostream>
#include <CudaUtils.h>
#include <Usutf.h>
#include <CudaUtils.h>
#include <EigenUtils.h>
#include <Eigen/Core>
#include <chrono>

CudaUtils::CuBlasHandle handle;

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

USUTF_TEST(testCudaUtils_gemm)
{
	Eigen::MatrixXf ae(2, 3);
	ae <<
	1.0f, 2.3f, -1.3f,
	4.5f, 2.2f, 11.4f;
	Eigen::MatrixXf be(3, 4);
	be <<
	1.0f,  2.3f, -1.3f,  3.9f,
	4.5f,  2.2f, 11.4f, -0.9f,
	1.3f, -0.2f,  2.4f,  3.1f;
	Eigen::MatrixXf c_ref = ae * be;
	CudaUtils::MatrixF a = CudaUtils::toDevice(EigenUtils::toMatrix(ae));
	CudaUtils::MatrixF b = CudaUtils::toDevice(EigenUtils::toMatrix(be));
	CudaUtils::MatrixF c_device = CudaUtils::allocateMatrixOnDeviceF(c_ref.rows(), c_ref.cols());
	CudaUtils::gemm(c_device, a, b, handle);
	CudaUtils::MatrixF c_host = CudaUtils::toHost(c_device);
	Eigen::MatrixXf c_test = EigenUtils::toEigen(c_host);
	Usutf::test(EigenUtils::almostEqual(c_test, c_ref));
}

USUTF_TEST(testCudaUtils_gemmTiled)
{
	Eigen::MatrixXf ae(16, 16), be(16, 16);
	ae.setRandom();
	be.setRandom();
	Eigen::MatrixXf c_ref = ae * be;
	CudaUtils::MatrixF a = CudaUtils::toDevice(EigenUtils::toMatrix(ae));
	CudaUtils::MatrixF b = CudaUtils::toDevice(EigenUtils::toMatrix(be));
	CudaUtils::MatrixF c_device = CudaUtils::allocateMatrixOnDeviceF(c_ref.rows(), c_ref.cols());
	CudaUtils::gemmTiled(c_device, a, b);
	CudaUtils::MatrixF c_host = CudaUtils::toHost(c_device);
	Eigen::MatrixXf c_test = EigenUtils::toEigen(c_host);
	Usutf::test(EigenUtils::almostEqual(c_test, c_ref));
}

USUTF_TEST(testCudaUtils_gemmTiled_2)
{
	Eigen::MatrixXf ae(256, 256), be(256, 256);
	ae.setRandom();
	be.setRandom();
	Eigen::MatrixXf c_ref = ae * be;
	CudaUtils::MatrixF a = CudaUtils::toDevice(EigenUtils::toMatrix(ae));
	CudaUtils::MatrixF b = CudaUtils::toDevice(EigenUtils::toMatrix(be));
	CudaUtils::MatrixF c_device = CudaUtils::allocateMatrixOnDeviceF(c_ref.rows(), c_ref.cols());
	CudaUtils::gemmTiled(c_device, a, b);
	CudaUtils::MatrixF c_host = CudaUtils::toHost(c_device);
	Eigen::MatrixXf c_test = EigenUtils::toEigen(c_host);
	Usutf::test(EigenUtils::almostEqual(c_test, c_ref));
}

USUTF_TEST(testCudaUtils_gemmTiled_3)
{
	Eigen::MatrixXf ae(96, 64), be(64, 32);
	ae.setRandom();
	be.setRandom();
	Eigen::MatrixXf c_ref = ae * be;
	CudaUtils::MatrixF a = CudaUtils::toDevice(EigenUtils::toMatrix(ae));
	CudaUtils::MatrixF b = CudaUtils::toDevice(EigenUtils::toMatrix(be));
	CudaUtils::MatrixF c_device = CudaUtils::allocateMatrixOnDeviceF(c_ref.rows(), c_ref.cols());
	CudaUtils::gemmTiled(c_device, a, b);
	CudaUtils::MatrixF c_host = CudaUtils::toHost(c_device);
	Eigen::MatrixXf c_test = EigenUtils::toEigen(c_host);
	Usutf::test(EigenUtils::almostEqual(c_test, c_ref, 1.0e-3f, 2));
}

USUTF_TEST(benchmark_gemmTiled)
{
	Eigen::MatrixXf ae(1024, 1024), be(1024, 64);
	ae.setRandom();
	be.setRandom();
	Eigen::MatrixXf c_ref = ae * be;
	CudaUtils::MatrixF a = CudaUtils::toDevice(EigenUtils::toMatrix(ae));
	CudaUtils::MatrixF b = CudaUtils::toDevice(EigenUtils::toMatrix(be));
	CudaUtils::MatrixF c_device = CudaUtils::allocateMatrixOnDeviceF(c_ref.rows(), c_ref.cols());
	CudaUtils::deviceSynchronize();
	auto t1 = std::chrono::high_resolution_clock::now();
	CudaUtils::gemmTiled(c_device, a, b);
	CudaUtils::deviceSynchronize();
	auto t2 = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
	CudaUtils::MatrixF c_host = CudaUtils::toHost(c_device);
	Eigen::MatrixXf c_test = EigenUtils::toEigen(c_host);
	Usutf::test(EigenUtils::almostEqual(c_test, c_ref, 1.0e-3f, 2));
	std::cout << "t2-t1 = " << time_span.count() << " seconds.";
}

int main(int argc, char *argv[])
{
	std::cout << "test lib double_block_sparse...\n";	
	int r = Usutf::runTests(argc, argv);
	return r;
}
