#include <iostream>
#include <CudaUtils.h>
#include <Usutf.h>
#include <CudaUtils.h>

USUTF_TEST(testDeviceAllocate)
{
	void *x = CudaUtils::deviceAllocate(1);
	Usutf::test(x != 0);
	CudaUtils::deviceFree(x);
}

int main(int argc, char *argv[])
{
	std::cout << "test lib double_block_sparse...\n";	
	int r = Usutf::runTests(argc, argv);
	return r;
}
