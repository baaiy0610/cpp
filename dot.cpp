#include<iostream>
#include<hip/hip_runtime.h>
#include<ctime>
#include<time.h>
using namespace std;

const int N=1024*1024;
const int threadPerBlock=1024;
const int blockPreGrid=1024;

__global__
void myKernel(int *a, int *b, int *c)
{
    __shared__ int cache[threadPerBlock];
    int tid = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x;
    int cacheIndex = hipThreadIdx_x;
    int temp=0;
    if (tid<N){
        temp+=a[tid]*b[tid];
    }
    cache[cacheIndex]=temp;
    __syncthreads();

    int i = hipBlockDim_x/2;
    while(i!=0){
        if (cacheIndex<i)
            cache[cacheIndex]+=cache[cacheIndex+i];
        __syncthreads();
        i/=2;
    }
    if (cacheIndex==0)
        c[hipBlockIdx_x]=cache[0];
}

int main(){
    clock_t start,end;
    srand(time(NULL));
    //initial data
    int *h_a = new int[N];
    int *h_b = new int[N];
    int c=0;
    int *partial_c = new int[blockPreGrid];
    for (int i=0;i<N;i++){
        h_a[i]=rand()%10;
        h_b[i]=rand()%10;
    }
    //time in the CPU
    int sum=0;
    start=clock();
    for (int i =0;i<N;i++){ 
        sum+=h_a[i]*h_b[i];
    }
    end=clock();
    cout<<"F1运行时间"<<(double)(end-start)/CLOCKS_PER_SEC<<endl;
    cout<<sum<<endl;

    //memory allocation in device
    int *d_a,*d_b,*d_partial_c;

    hipMalloc(&d_a,sizeof(int)*N);
    hipMalloc(&d_b,sizeof(int)*N);
    hipMalloc(&d_partial_c,sizeof(int)*blockPreGrid);
    //copy data from host to device
    hipMemcpy(d_a,h_a,sizeof(int)*N,hipMemcpyHostToDevice);
    hipMemcpy(d_b,h_b,sizeof(int)*N,hipMemcpyHostToDevice);
    //launch kernel
    start=clock();
    hipLaunchKernelGGL(myKernel,dim3(blockPreGrid),dim3(threadPerBlock),0,0,d_a,d_b,d_partial_c);
    end=clock();
    cout<<"F2运行时间"<<(double)(end-start)/CLOCKS_PER_SEC<<endl;
    //return the result to host
    hipMemcpy(partial_c,d_partial_c,sizeof(int)*blockPreGrid,hipMemcpyDeviceToHost);
    for (int i=0;i<blockPreGrid;i++)
    {
        c+=partial_c[i];
    }
    cout<<c<<endl;

    hipFree(d_a);
    hipFree(d_b);
    hipFree(d_partial_c);

    delete[] h_a;
    delete[] h_b;
    delete[] partial_c;
 
    // for (int i= 0;i<5;i++){
    //     cout<<h_a[i]<<" "<<h_b[i]<<endl;
    // }
    return 0;
}
