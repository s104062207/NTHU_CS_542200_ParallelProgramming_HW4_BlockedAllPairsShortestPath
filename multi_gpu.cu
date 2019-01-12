#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <cuda_runtime.h>
#include <omp.h>
struct timespec s, e, all_start, all_end;
long cpu_time = 0, io_time = 0, communication_time = 0;
void calc_time(long* target, struct timespec a, struct timespec b)
{
    int sec = a.tv_sec - b.tv_sec;
    int nsec = a.tv_nsec - b.tv_nsec;
    *target += ((long) sec) * 1000000000 + nsec;
}
__global__ 
void calPhase1(int B, int Round, int* Dist, int node, int pitch)
{
	extern __shared__ int sdata[];
	int x = threadIdx.x;
    int y = threadIdx.y;
	int sx = Round*B+x;
	int sy = Round*B+y;
	
	sdata[x*B+y]=Dist[sx*pitch+sy];
	__syncthreads();
	int tem;
	#pragma unroll
	for (int k = 0; k < B ; ++k) 
	{		
		tem=sdata[x*B+k] + sdata[k*B+y];
		if (tem < sdata[x*B+y])
			sdata[x*B+y] = tem;	
		__syncthreads();
	}
	Dist[sx*pitch+sy]=sdata[x*B+y];
	__syncthreads();
}
__global__ 
void calPhase2(int B, int Round, int* Dist, int node, int pitch)
{
	if(blockIdx.x==Round)
		return;
	extern __shared__ int sm[];
	int* p = &sm[B*B];
	
	int x = threadIdx.x;
    int y = threadIdx.y;
	
	unsigned int sx = Round*B+x;
	unsigned int sy = Round*B+y;	
	sm[x*B+y]=Dist[sx*pitch+sy];
	
	unsigned int rx = blockIdx.x*B+x;
	unsigned int cy = blockIdx.x*B+y;
	unsigned int idx= (blockIdx.y == 1)?rx*pitch+sy:sx*pitch+cy;
	p[x*B+y]=Dist[idx];
	__syncthreads();
	
	int* a =(blockIdx.y == 0)?&sm[0]:p;
	int* b =(blockIdx.y == 1)?&sm[0]:p;
	int tem;
	#pragma unroll
	for (int k = 0; k < B ; ++k) 
	{
		tem=a[x*B+k] + b[k*B+y];
		if ( tem < p[x*B+y])
			p[x*B+y] = tem;
	}
	Dist[idx]=p[x*B+y];
}
__global__ 
void calPhase3(int B, int Round, int* Dist, int node, int pitch,int threadId,int halfRound)
{
	int blockIdxx=blockIdx.x;
	if(threadId==1)
		blockIdxx=blockIdxx+halfRound;
	if (blockIdxx == Round || blockIdx.y == Round) 
		return;
	extern __shared__ int sm[];
	int* pr = &sm[0];
	int* pc = &sm[B*B];
	
	int x = threadIdx.x;
    int y = threadIdx.y;
	
	int rx = blockIdxx*blockDim.x+x;
	int ry = Round*B+y;
	
	int cx = Round*B+x;
	int cy = blockIdx.y*blockDim.y+y;
	
	pr[x*B+y]=Dist[rx*pitch+ry];
	pc[x*B+y]=Dist[cx*pitch+cy];
	__syncthreads();
	
	if (rx >= node || cy >= node) 
		return;
	
	int tem;
	int ans=Dist[rx*pitch+cy] ;
	#pragma omp parallel for 
	for (int k = 0; k < B ; ++k) {		
		tem=pr[x*B+k] + pc[k*B+y];
		if ( tem<ans){
			ans=tem;
		}
	}
	Dist[rx*pitch+cy] = ans;
}
int main(int argc, char* argv[])
{
	
	
	//input(argv[1]);
	FILE *fp = fopen(argv[1], "rb");
	int n, edge;
	clock_gettime(CLOCK_REALTIME, &s);
	fread(&n   , sizeof(int), 1, fp); 
	fread(&edge, sizeof(int), 1, fp);
	clock_gettime(CLOCK_REALTIME, &e);
    calc_time(&io_time, e, s);
	
	int B = (n>32)?32:16;//atoi(argv[3]);
	int round = (n + B -1)/B;
	int pitch_n = round*B;//(n%B==0)?n:n-n%B+B;
	int* Dist;//=(int*) malloc(pitch_n * pitch_n * sizeof(int));
    cudaMallocHost(&Dist, sizeof(int)*pitch_n*pitch_n);
	clock_gettime(CLOCK_REALTIME, &s);
	#pragma omp parallel for collapse(2)
	for (int i = 0; i < pitch_n; ++i) {
		for (int j = 0; j < pitch_n; ++j) {
			if (i == j)	
				Dist[i*pitch_n+j] = 0;
			else		
				Dist[i*pitch_n+j] = 1000000000;
		}
	}
	    clock_gettime(CLOCK_REALTIME, &e);
    calc_time(&cpu_time, e, s);
	int* temp =(int*) malloc(edge * 3 * sizeof(int));
	clock_gettime(CLOCK_REALTIME, &s);
    fread(temp, sizeof(int), edge * 3, fp);
	    clock_gettime(CLOCK_REALTIME, &e);
    calc_time(&io_time, e, s);
	
	
	clock_gettime(CLOCK_REALTIME, &s);
    #pragma omp parallel for 
    for (int i = 0; i < edge*3; i=i+3) 
        Dist[temp[i]*pitch_n+temp[i+1]] = temp[i+2];
	    clock_gettime(CLOCK_REALTIME, &e);
    calc_time(&cpu_time, e, s);
	
	//block_FW(B);
	float time;
    float GPU_time = 0;
    cudaEvent_t start, stop;
    cudaEventCreate (&start);
    cudaEventCreate (&stop);
	int* devDist[2];
	#pragma omp parallel num_threads(2)
    {
		int threadId = omp_get_thread_num();
		//cudaSetDevice(threadId);
		cudaMalloc(&devDist[threadId], sizeof(int) * pitch_n * pitch_n);
		cudaMemcpy(devDist[threadId], Dist, sizeof(int) * pitch_n * pitch_n, cudaMemcpyHostToDevice);
	
		dim3 grid1(1, 1);
		dim3 grid2(round, 2);
		dim3 grid3(round, round);
		
		if(threadId == 0) grid3.x = round/2;
		else grid3.x = round-(round/2);
		
		dim3 block(B, B);
		int sSize = B * B * sizeof(int);
		cudaEventRecord (start, 0);
		for (int r = 0; r < round; ++r) {
			calPhase1<<<grid1, block, sSize  >>>(B, r, devDist[threadId], n, pitch_n);
			calPhase2<<<grid2, block, sSize*2>>>(B, r, devDist[threadId], n, pitch_n);
			calPhase3<<<grid3, block, sSize*2>>>(B, r, devDist[threadId], n, pitch_n,threadId,round/2);
			cudaDeviceSynchronize();
			clock_gettime(CLOCK_REALTIME, &s);
			#pragma omp barrier
			if(threadId==0)
				cudaMemcpyPeer(devDist[1], 1, devDist[0], 0, round/2*B*sizeof(int) * pitch_n);
			else
				cudaMemcpyPeer(&devDist[0][round/2 *pitch_n*B], 0, &devDist[1][round/2 *pitch_n*B], 1, (round-round/2)*B*sizeof(int) * pitch_n);
			#pragma omp barrier
			clock_gettime(CLOCK_REALTIME, &e);
            calc_time(&communication_time, e, s);
		}
		
	}
	cudaEventRecord (stop, 0);
	cudaEventElapsedTime (&time, start, stop);
	GPU_time = time/1000 - (communication_time/1000000000.0);
	cudaDeviceSynchronize();
	cudaMemcpy2D(Dist, sizeof(int) *n, devDist[0], sizeof(int) * pitch_n, sizeof(int) *n, n, cudaMemcpyDeviceToHost);
	
	//output(argv[2]);
	fp = fopen(argv[2], "wb+");
	clock_gettime(CLOCK_REALTIME, &s);
	fwrite(Dist, sizeof(int), n*n, fp);
	clock_gettime(CLOCK_REALTIME, &e);
	calc_time(&io_time, e, s);
	
	printf("io_time:%lf\n",(io_time/1000000000.0));
	printf ("GPU time = %lf\n", GPU_time);
	printf("communication time = %lf\n",(communication_time/1000000000.0));
	printf("cpu_time:%lf\n",(cpu_time/1000000000.0));
	fclose(fp);

	return 0;
}
