#include <vector_types.h>
#include <cuda.h>

#define CEIL_DIV(x, y) (((x) + (y) -   1) / (y))

__device__ __forceinline__ float  to_float(float  v) { return v; }
__device__ __forceinline__ float4 to_float(float4 v) { return v; }

__device__ __forceinline__ void ew_zero(float  &a) { a = 0.0f; }
__device__ __forceinline__ void ew_zero(float4 &a) { a.x = a.y = a.z = a.w = 0.0f; }

__device__ __forceinline__ void ld_shared_v4(int a, float* v)
{
    asm volatile ("ld.shared.v4.f32 {%0, %1, %2, %3}, [%4];"  : "=f"(v[0]),"=f"(v[1]),"=f"(v[2]),"=f"(v[3]) : "r"(a));
}

__device__ __forceinline__  void st_shared_v4(int a, float4 v)
{
    asm volatile ("st.shared.v4.f32 [%0], {%1, %2, %3, %4};" :: "r"(a), "f"(v.x),"f"(v.y),"f"(v.z),"f"(v.w) );
}

__device__ __forceinline__  void st_shared_v4(int a, float* v)
{
    asm volatile ("st.shared.v4.f32 [%0], {%1, %2, %3, %4};" :: "r"(a), "f"(v[0]),"f"(v[1]),"f"(v[2]),"f"(v[3]) );
}

__device__ __forceinline__ void atomicRed(float *ptr, float val, int i=0, bool b=true)
{
    if (b)
        //asm volatile ("red.global.add.f32 [%0], %1;" :: "l"(ptr+i), "f"(val)  );
        asm volatile ("st.global.f32 [%0], %1;" :: "l"(ptr+i), "f"(val)  );
}
__device__ __forceinline__ void atomicRed(float4 *ptr, float4 val, int i=0, bool b=true)
{
    if (b)
        //asm volatile ("red.global.add.f32 [%0], %1;" :: "l"(ptr+i), "f"(val.x)  );
        asm volatile ("st.global.f32 [%0], %1;" :: "l"(ptr+i), "f"(val.x)  );
}

#define ADD_PTR_U(T) \
__device__ __forceinline__ const T* add_ptr_u(const T* src, int offset)      \
{                                                                            \
    const T* dst;                                                            \
    asm("{                       \n\t"                                       \
        ".reg .u32 lo,hi,of;     \n\t"                                       \
        "mul.lo.u32 of, %2, %3;  \n\t"                                       \
        "mov.b64    {lo,hi}, %1; \n\t"                                       \
        "add.cc.u32  lo,lo,  of; \n\t"                                       \
        "addc.u32    hi,hi,  0;  \n\t"                                       \
        "mov.b64 %0, {lo,hi};    \n\t"                                       \
        "}" : "=l"(dst) : "l"(src), "r"(offset), "r"((int)sizeof(*src)));    \
    return dst;                                                              \
}                                                                            \
__device__ __forceinline__ T* add_ptr_u(T* src, int offset)                  \
{                                                                            \
    T* dst;                                                                  \
    asm("{                       \n\t"                                       \
        ".reg .u32 lo,hi,of;     \n\t"                                       \
        "mul.lo.u32 of, %2, %3;  \n\t"                                       \
        "mov.b64    {lo,hi}, %1; \n\t"                                       \
        "add.cc.u32  lo,lo,  of; \n\t"                                       \
        "addc.u32    hi,hi,  0;  \n\t"                                       \
        "mov.b64 %0, {lo,hi};    \n\t"                                       \
        "}" : "=l"(dst) : "l"(src), "r"(offset), "r"((int)sizeof(*src)));    \
    return dst;                                                              \
}

ADD_PTR_U(float )
ADD_PTR_U(float4)

ADD_PTR_U(unsigned char)
ADD_PTR_U(unsigned short)
ADD_PTR_U(unsigned int)
ADD_PTR_U(int)
