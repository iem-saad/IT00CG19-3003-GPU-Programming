/*******************************************************
 * GPU PROJECT CODE                                   *
 *                                                    *
 * Author: Saad Abdullah                              *
 * E-mail: saad.abdullah@abo.fi                       *
 * Created: 18 JAN 2025                               *
 * Last modification: 26 JAN 2025                     *
 *******************************************************/

#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

#define totaldegrees 180
#define binsperdegree 4
#define threadsperblock 512
#define THREAD_WORK_SIZE 8

// data for the real galaxies will be read into these arrays
float *ra_real, *decl_real;
// number of real galaxies
int    NoofReal;

// data for the simulated random galaxies will be read into these arrays
float *ra_sim, *decl_sim;
// number of simulated random galaxies
int    NoofSim;

unsigned int *histogramDR, *histogramDD, *histogramRR;
float *d_histogram;

// CUDA Kernel to Convert Arcminutes to Radians
__global__ void convertToRadiansKernel(int numGalaxies, float *ra, float *decl) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numGalaxies) {
    ra[idx] = ra[idx] / 60.0 * M_PI / 180.0;
    decl[idx] = decl[idx] / 60.0 * M_PI / 180.0;
  }
}

// Device function to compute cosTheta
__device__ float computeCosTheta(float decl1, float decl2, float ra1, float ra2) {
  return cosf(decl1) * cosf(decl2) * cosf(ra1 - ra2) +
         sinf(decl1) * sinf(decl2);
}

// Device function to initialize block histogram
__device__ void initializeBlockHistogram(unsigned int *blockHistogram, int totalBins, int threadsPerBlock) {
  for (int i = threadIdx.x; i < totalBins; i += threadsPerBlock) {
    blockHistogram[i] = 0;
  }
  __syncthreads();
}

// Device function to load data into shared memory
__device__ void loadSharedMemory(float *globalRa, float *globalDecl,
                                float *sharedRa, float *sharedDecl,
                                int numGalaxies, int blockStart, int threadsPerBlock) {
  int index = blockStart + threadIdx.x;
  if (index < numGalaxies) {
    sharedRa[threadIdx.x] = globalRa[index];
    sharedDecl[threadIdx.x] = globalDecl[index];
  } else {
    sharedRa[threadIdx.x] = 0.0f;
    sharedDecl[threadIdx.x] = 0.0f;
  }
  __syncthreads();
}

// Device function to process galaxy pairs
__device__ void processGalaxyPairs(float *ra1_Shared, float *decl1_Shared,
                                  float ra2, float decl2, unsigned int *blockHistogram,
                                  int blockSize, int blockStart, int numGalaxies,
                                  int binsPerDegree, int totalBins) {
  for (int i = 0; i < blockSize && (blockStart + i) < numGalaxies; i++) {
    float cosTheta = computeCosTheta(decl1_Shared[i], decl2, ra1_Shared[i], ra2);

    if (cosTheta > 1.0f){
      cosTheta = 1.0f;
    }
    else if (cosTheta < -1.0f){
      cosTheta = -1.0f;
    }

    float degree = acosf(cosTheta) * 180.0f / M_PI;

    int bin = (int)(degree * binsPerDegree);
    if (bin >= 0 && bin < totalBins) {
      atomicAdd(&blockHistogram[bin], 1);
    }
  }
}

// Kernel to compute histogram using shared memory
__global__ void calculateHistograms(float *ra_1, float *decl_1, float *ra_2, float *decl_2,
                                    unsigned int *globalHistogram, int numGalaxies,
                                    int binsPerDegree, int totalBins) {
  __shared__ float ra1_Shared[threadsperblock];
  __shared__ float decl1_Shared[threadsperblock];
  __shared__ unsigned int blockHistogram[totaldegrees * binsperdegree];

  int threadId = threadIdx.x + blockIdx.x * blockDim.x;

  initializeBlockHistogram(blockHistogram, totalBins, blockDim.x);

  for (int blockStart = 0; blockStart < numGalaxies; blockStart += blockDim.x) {
    loadSharedMemory(ra_1, decl_1, ra1_Shared, decl1_Shared, numGalaxies, blockStart, blockDim.x);

    if (threadId < numGalaxies) {
      float ra2 = ra_2[threadId];
      float decl2 = decl_2[threadId];
      processGalaxyPairs(ra1_Shared, decl1_Shared, ra2, decl2,
                         blockHistogram, blockDim.x, blockStart, numGalaxies, binsPerDegree, totalBins);
    }
    __syncthreads();
  }

  // Aggregate block histogram into global memory
  for (int i = threadIdx.x; i < totalBins; i += blockDim.x) {
    atomicAdd(&globalHistogram[i], blockHistogram[i]);
  }
}

__global__ void computeOmegaKernel(const unsigned int *histogramDD,
                                   const unsigned int *histogramDR,
                                   const unsigned int *histogramRR,
                                   float *omega,
                                   int totalBins) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // Ensure the thread operates within valid bounds
  if (idx < totalBins) {
    unsigned int rr = histogramRR[idx];
    if (rr > 0) {
      omega[idx] = ((float)histogramDD[idx] - 2.0f * (float)histogramDR[idx] + (float)rr) / (float)rr;
    } else {
      omega[idx] = 0.0f;
    }
  }
}

int main(int argc, char *argv[])
{
   int    readdata(char *argv1, char *argv2);
   int    getDevice(int deviceno);
   double start, end, kerneltime;
   struct timeval _ttime;
   struct timezone _tzone;

   FILE *outfile;

   if ( argc != 4 ) {printf("Usage: a.out real_data random_data output_data\n");return(-1);}

   if ( getDevice(0) != 0 ) return(-1);

   if ( readdata(argv[1], argv[2]) != 0 ) return(-1);

  cudaMallocManaged((void ** ) & histogramDR, totaldegrees * binsperdegree * sizeof(unsigned int));
  cudaMallocManaged((void ** ) & histogramDD, totaldegrees * binsperdegree * sizeof(unsigned int));
  cudaMallocManaged((void ** ) & histogramRR, totaldegrees * binsperdegree * sizeof(unsigned int));
  
  // start time
  kerneltime = 0.0;
  gettimeofday(&_ttime, &_tzone);
  start = (double)_ttime.tv_sec + (double)_ttime.tv_usec/1000000.;

  int totalThreads = (NoofReal / THREAD_WORK_SIZE) * (NoofReal / THREAD_WORK_SIZE);
  int numBlocks = (totalThreads + threadsperblock - 1) / threadsperblock;

  calculateHistograms<<<numBlocks, threadsperblock>>>(
    ra_real, decl_real, ra_sim, decl_sim, histogramDR, NoofReal, binsperdegree, totaldegrees * binsperdegree);

  calculateHistograms<<<numBlocks, threadsperblock>>>(
    ra_real, decl_real, ra_real, decl_real, histogramDD, NoofReal, binsperdegree, totaldegrees * binsperdegree);

  calculateHistograms<<<numBlocks, threadsperblock>>>(
    ra_sim, decl_sim, ra_sim, decl_sim, histogramRR, NoofReal, binsperdegree, totaldegrees * binsperdegree);

  cudaDeviceSynchronize();


  float *omega;
  cudaMallocManaged(&omega, totaldegrees * binsperdegree * sizeof(float));

  int blocksPerGrid = (totaldegrees * binsperdegree + threadsperblock - 1) / threadsperblock;
  computeOmegaKernel<<<blocksPerGrid, threadsperblock>>>(histogramDD, histogramDR, histogramRR, omega, totaldegrees * binsperdegree);

  // Synchronize to ensure kernel execution is complete
  cudaDeviceSynchronize();

  outfile = fopen(argv[3], "w");
  if (outfile == NULL) {
    perror("Error opening the file");
    return 1;
  }
  printf("\n*****************************************************************************\n");
  printf("  %5s-%5s      %10s      %10s      %10s      %10s\n", 
         "Bin", "", "Omega", "histogramDD", "histogramDR", "histogramRR");
  printf("=============================================================================\n");
  for (int i = 0; i < 6; i++) {
    float binStart = i * 0.25;
    float binEnd = (i + 1) * 0.25;
    printf("  %5.2f-%5.2f      %10.6f      %10d      %10d      %10d\n", 
           binStart, binEnd, omega[i], histogramDD[i], histogramDR[i], histogramRR[i]);
  }
  printf("=============================================================================\n");
  
  unsigned long long sumDD = 0, sumDR = 0, sumRR = 0;
  fprintf(outfile, "Bin,Omega,HistogramDD,HistogramDR,HistogramRR\n");
  for (int i = 0; i < 720; i++) {
    sumDD += histogramDD[i];
    sumDR += histogramDR[i];
    sumRR += histogramRR[i];
    float binStart = i * 0.25;
    float binEnd = (i + 1) * 0.25;
    fprintf(outfile, "%.2f-%.2f,%f,%d,%d,%d\n", 
            binStart, binEnd, omega[i], histogramDD[i], histogramDR[i], histogramRR[i]);
  }
  fclose(outfile);
  // end time
  gettimeofday(&_ttime, &_tzone);
  end = (double)_ttime.tv_sec + (double)_ttime.tv_usec/1000000.;
  kerneltime += end-start;

  printf("=============================================================================\n");
  printf("Sum of all histogramDD values: %llu\n", sumDD);
  printf("Sum of all histogramDR values: %llu\n", sumDR);
  printf("Sum of all histogramRR values: %llu\n", sumRR);
  printf("=============================================================================\n");
  printf("   Time in GPU    : %f seconds\n", kerneltime);
  printf("   Results saved  : %s\n", argv[3]);
  printf("*****************************************************************************\n");

  cudaFree(ra_real);
  cudaFree(decl_real);
  cudaFree(ra_sim);
  cudaFree(decl_sim);
  cudaFree(histogramDR);
  cudaFree(histogramDD);
  cudaFree(histogramRR);
  cudaFree(omega);

  return(0);
}


int readdata(char *argv1, char *argv2)
{
  int i,linecount;
  char inbuf[180];
  double ra, dec;
  FILE *infil;
                                         
  printf("   Assuming input data is given in arc minutes!\n");
                          // spherical coordinates phi and theta in radians:
                          // phi   = ra/60.0 * dpi/180.0;
                          // theta = (90.0-dec/60.0)*dpi/180.0;

  infil = fopen(argv1,"r");
  if ( infil == NULL ) {printf("Cannot open input file %s\n",argv1);return(-1);}

  // read the number of galaxies in the input file
  int announcednumber;
  if ( fscanf(infil,"%d\n",&announcednumber) != 1 ) {printf(" cannot read file %s\n",argv1);return(-1);}
  linecount =0;
  while ( fgets(inbuf,180,infil) != NULL ) ++linecount;
  rewind(infil);

  if ( linecount == announcednumber ) printf("   %s contains %d galaxies\n",argv1, linecount);
  else {
    printf("   %s does not contain %d galaxies but %d\n",argv1, announcednumber,linecount);
    return(-1);
  }

  NoofReal = linecount;
  // https://www.olcf.ornl.gov/wp-content/uploads/2019/06/06_Managed_Memory.pdf
  // Loading in Unified Memory, so that its accessible by both CPU and GPU.
  cudaMallocManaged((void ** ) & ra_real, NoofReal * sizeof(float));
  cudaMallocManaged((void ** ) & decl_real, NoofReal * sizeof(float));

  // skip the number of galaxies in the input file
  if ( fgets(inbuf,180,infil) == NULL ) return(-1);
  i = 0;
  while ( fgets(inbuf,80,infil) != NULL ){
    if ( sscanf(inbuf,"%lf %lf",&ra,&dec) != 2 ) {
      printf("   Cannot read line %d in %s\n",i+1,argv1);
      fclose(infil);
      return(-1);
    }
    ra_real[i]   = (float)ra;
    decl_real[i] = (float)dec;
    ++i;
  }

  fclose(infil);

  if ( i != NoofReal ){
    printf("   Cannot read %s correctly\n",argv1);
    return(-1);
  }

  infil = fopen(argv2,"r");
  if ( infil == NULL ) {printf("Cannot open input file %s\n",argv2);return(-1);}

  if ( fscanf(infil,"%d\n",&announcednumber) != 1 ) {printf(" cannot read file %s\n",argv2);return(-1);}
  linecount =0;
  while ( fgets(inbuf,80,infil) != NULL ) ++linecount;
  rewind(infil);

  if ( linecount == announcednumber ) printf("   %s contains %d galaxies\n",argv2, linecount);
  else{
    printf("   %s does not contain %d galaxies but %d\n",argv2, announcednumber,linecount);
    return(-1);
  }

  NoofSim = linecount;
  cudaMallocManaged((void ** ) & ra_sim, NoofSim * sizeof(float));
  cudaMallocManaged((void ** ) & decl_sim, NoofSim * sizeof(float));

  // skip the number of galaxies in the input file
  if ( fgets(inbuf,180,infil) == NULL ) return(-1);
  i =0;
  while ( fgets(inbuf,80,infil) != NULL ){
    if ( sscanf(inbuf,"%lf %lf",&ra,&dec) != 2 ){
      printf("   Cannot read line %d in %s\n",i+1,argv2);
      fclose(infil);
      return(-1);
    }
    ra_sim[i]   = (float)ra;
    decl_sim[i] = (float)dec;
    ++i;
  }

  fclose(infil);

  if ( i != NoofSim ) {
    printf("   Cannot read %s correctly\n",argv2);
    return(-1);
  }

  // Convert Arcminutes to Radians Using GPU
  int blocksReal = (NoofReal + threadsperblock - 1) / threadsperblock;
  int blocksSim = (NoofSim + threadsperblock - 1) / threadsperblock;

  convertToRadiansKernel<<<blocksReal, threadsperblock>>>(NoofReal, ra_real, decl_real);
  convertToRadiansKernel<<<blocksSim, threadsperblock>>>(NoofSim, ra_sim, decl_sim);

  // Synchronize to ensure the GPU computation is complete
  cudaDeviceSynchronize();

  return(0);
}


int getDevice(int deviceNo)
{
  int deviceCount;
  cudaGetDeviceCount(&deviceCount);
  printf("   Found %d CUDA devices\n",deviceCount);
  if ( deviceCount < 0 || deviceCount > 128 ) return(-1);
  int device;
  for (device = 0; device < deviceCount; ++device) {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);
    printf("      Device %s                  device %d\n", deviceProp.name,device);
    printf("         compute capability            =        %d.%d\n", deviceProp.major, deviceProp.minor);
    printf("         totalGlobalMemory             =       %.2lf GB\n", deviceProp.totalGlobalMem/1000000000.0);
    printf("         l2CacheSize                   =   %8d B\n", deviceProp.l2CacheSize);
    printf("         regsPerBlock                  =   %8d\n", deviceProp.regsPerBlock);
    printf("         multiProcessorCount           =   %8d\n", deviceProp.multiProcessorCount);
    printf("         maxThreadsPerMultiprocessor   =   %8d\n", deviceProp.maxThreadsPerMultiProcessor);
    printf("         sharedMemPerBlock             =   %8d B\n", (int)deviceProp.sharedMemPerBlock);
    printf("         warpSize                      =   %8d\n", deviceProp.warpSize);
    printf("         clockRate                     =   %8.2lf MHz\n", deviceProp.clockRate/1000.0);
    printf("         maxThreadsPerBlock            =   %8d\n", deviceProp.maxThreadsPerBlock);
    printf("         asyncEngineCount              =   %8d\n", deviceProp.asyncEngineCount);
    printf("         f to lf performance ratio     =   %8d\n", deviceProp.singleToDoublePrecisionPerfRatio);
    printf("         maxGridSize                   =   %d x %d x %d\n",
                      deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
    printf("         maxThreadsDim in thread block =   %d x %d x %d\n",
                      deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
    printf("         concurrentKernels             =   ");
    if(deviceProp.concurrentKernels==1) printf("     yes\n"); else printf("    no\n");
    printf("         deviceOverlap                 =   %8d\n", deviceProp.deviceOverlap);
    if(deviceProp.deviceOverlap == 1)
    printf("            Concurrently copy memory/execute kernel\n");
  }

  cudaSetDevice(deviceNo);
  cudaGetDevice(&device);
  if ( device != 0 ) printf("   Unable to set device 0, using %d instead",device);
  else printf("   Using CUDA device %d\n\n", device);

  return(0);
}
