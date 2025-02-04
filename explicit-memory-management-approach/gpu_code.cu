/*******************************************************
 * GPU PROJECT CODE                                   *
 *                                                    *
 * Author: Saad Abdullah                              *
 * E-mail: saad.abdullah@abo.fi                       *
 * Created: 03 JAN 2025                               *
 * Last modification: 27 JAN 2025                     *
 *******************************************************/

#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

#define totaldegrees 180
#define binsperdegree 4
#define threadsperblock 512

// data for the real galaxies will be read into these arrays
float *ra_real, *decl_real;
float *d_ra_real, *d_decl_real;
int NoofReal;

// data for the simulated random galaxies will be read into these arrays
float *ra_sim, *decl_sim;
float *d_ra_sim, *d_decl_sim;
int NoofSim;

unsigned int *histogramDR, *histogramDD, *histogramRR;
unsigned int *d_histogramDR, *d_histogramDD, *d_histogramRR;
float *omega;
float *d_omega;

__global__ void convertToRadiansKernel(int numGalaxies, float *ra, float *decl) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numGalaxies) {
    ra[idx] = ra[idx] / 60.0 * M_PI / 180.0;
    decl[idx] = decl[idx] / 60.0 * M_PI / 180.0;
  }
}

__device__ float computeCosTheta(float decl1, float decl2, float ra1, float ra2) {
  return cosf(decl1) * cosf(decl2) * cosf(ra1 - ra2) +
         sinf(decl1) * sinf(decl2);
}

__global__ void calculateHistograms(float *ra_1, float *decl_1, float *ra_2, float *decl_2,
                                    unsigned int *globalHistogram, int numGalaxies1, int numGalaxies2,
                                    int binsPerDegree, int totalBins) {
  __shared__ unsigned int blockHistogram[totaldegrees * binsperdegree];

  int threadId = threadIdx.x + blockIdx.x * blockDim.x;

  for (int i = threadIdx.x; i < totalBins; i += blockDim.x) {
    blockHistogram[i] = 0;
  }
  __syncthreads();

  if (threadId < numGalaxies1) {
    float ra1 = ra_1[threadId];
    float decl1 = decl_1[threadId];
    for (int j = 0; j < numGalaxies2; ++j) {
      float ra2 = ra_2[j];
      float decl2 = decl_2[j];
      float cosTheta = computeCosTheta(decl1, decl2, ra1, ra2);
      cosTheta = fmaxf(fminf(cosTheta, 1.0f), -1.0f);
      float degree = acosf(cosTheta) * 180.0f / M_PI;
      int bin = (int)(degree * binsPerDegree);

      if (bin < totalBins) {
        atomicAdd(&blockHistogram[bin], 1);
      }
    }
  }
  __syncthreads();

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

  kerneltime = 0.0;
  gettimeofday(&_ttime, &_tzone);
  start = (double)_ttime.tv_sec + (double)_ttime.tv_usec/1000000.;

  int totalBins = totaldegrees * binsperdegree;
  size_t histogramSize = totalBins * sizeof(unsigned int);
  size_t omegaSize = totalBins * sizeof(float);

  if (readdata(argv[1], argv[2]) != 0) return -1;

  cudaMalloc((void **)&d_histogramDR, histogramSize);
  cudaMalloc((void **)&d_histogramDD, histogramSize);
  cudaMalloc((void **)&d_histogramRR, histogramSize);
  cudaMalloc((void **)&d_omega, omegaSize);

  cudaMemset(d_histogramDR, 0, histogramSize);
  cudaMemset(d_histogramDD, 0, histogramSize);
  cudaMemset(d_histogramRR, 0, histogramSize);

  int blocksReal = (NoofReal + threadsperblock - 1) / threadsperblock;
  int blocksSim = (NoofSim + threadsperblock - 1) / threadsperblock;

  convertToRadiansKernel<<<blocksReal, threadsperblock>>>(NoofReal, d_ra_real, d_decl_real);
  convertToRadiansKernel<<<blocksSim, threadsperblock>>>(NoofSim, d_ra_sim, d_decl_sim);
  cudaDeviceSynchronize();

  calculateHistograms<<<blocksReal, threadsperblock>>>(d_ra_real, d_decl_real, d_ra_sim, d_decl_sim, d_histogramDR, NoofReal, NoofSim, binsperdegree, totalBins);
  calculateHistograms<<<blocksReal, threadsperblock>>>(d_ra_real, d_decl_real, d_ra_real, d_decl_real, d_histogramDD, NoofReal, NoofReal, binsperdegree, totalBins);
  calculateHistograms<<<blocksSim, threadsperblock>>>(d_ra_sim, d_decl_sim, d_ra_sim, d_decl_sim, d_histogramRR, NoofSim, NoofSim, binsperdegree, totalBins);
  cudaDeviceSynchronize();

  int omegaBlocks = (totalBins + threadsperblock - 1) / threadsperblock;
  computeOmegaKernel<<<omegaBlocks, threadsperblock>>>(d_histogramDD, d_histogramDR, d_histogramRR, d_omega, totalBins);
  cudaDeviceSynchronize();

  histogramDR = (unsigned int *)malloc(histogramSize);
  histogramDD = (unsigned int *)malloc(histogramSize);
  histogramRR = (unsigned int *)malloc(histogramSize);
  omega = (float *)malloc(omegaSize);

  cudaMemcpy(histogramDR, d_histogramDR, histogramSize, cudaMemcpyDeviceToHost);
  cudaMemcpy(histogramDD, d_histogramDD, histogramSize, cudaMemcpyDeviceToHost);
  cudaMemcpy(histogramRR, d_histogramRR, histogramSize, cudaMemcpyDeviceToHost);
  cudaMemcpy(omega, d_omega, omegaSize, cudaMemcpyDeviceToHost);

  outfile = fopen(argv[3], "w");
  if (!outfile) {
    perror("Error opening the file");
    return 1;
  }

  unsigned long long sumDD = 0, sumDR = 0, sumRR = 0;
  fprintf(outfile, "Bin,Omega,HistogramDD,HistogramDR,HistogramRR\n");
  for (int i = 0; i < totalBins; i++) {
    sumDD += histogramDD[i];
    sumDR += histogramDR[i];
    sumRR += histogramRR[i];
    float binStart = i * 0.25;
    float binEnd = (i + 1) * 0.25;
    fprintf(outfile, "%.2f-%.2f,%f,%d,%d,%d\n", 
            binStart, binEnd, omega[i], histogramDD[i], histogramDR[i], histogramRR[i]);
  }
  fclose(outfile);

  gettimeofday(&_ttime, &_tzone);
  end = (double)_ttime.tv_sec + (double)_ttime.tv_usec/1000000.;
  kerneltime += end-start;
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
  printf("=============================================================================\n");
  printf("Sum of all histogramDD values: %llu\n", sumDD);
  printf("Sum of all histogramDR values: %llu\n", sumDR);
  printf("Sum of all histogramRR values: %llu\n", sumRR);
  printf("=============================================================================\n");
  printf("   Time in GPU    : %f seconds\n", kerneltime);
  printf("   Results saved  : %s\n", argv[3]);
  printf("*****************************************************************************\n");

  free(ra_real);
  free(decl_real);
  free(ra_sim);
  free(decl_sim);
  free(histogramDR);
  free(histogramDD);
  free(histogramRR);
  free(omega);

  cudaFree(d_ra_real);
  cudaFree(d_decl_real);
  cudaFree(d_ra_sim);
  cudaFree(d_decl_sim);
  cudaFree(d_histogramDR);
  cudaFree(d_histogramDD);
  cudaFree(d_histogramRR);
  cudaFree(d_omega);

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
  else{
    printf("   %s does not contain %d galaxies but %d\n",argv1, announcednumber,linecount);
    return(-1);
  }

  NoofReal = linecount;
  ra_real   = (float *)calloc(NoofReal,sizeof(float));
  decl_real = (float *)calloc(NoofReal,sizeof(float));

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

  if ( i != NoofReal ) {
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
  ra_sim   = (float *)calloc(NoofSim,sizeof(float));
  decl_sim = (float *)calloc(NoofSim,sizeof(float));

  // skip the number of galaxies in the input file
  if ( fgets(inbuf,180,infil) == NULL ) return(-1);
  i =0;
  while ( fgets(inbuf,80,infil) != NULL ){
    if ( sscanf(inbuf,"%lf %lf",&ra,&dec) != 2 ) {
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

  cudaMalloc((void **)&d_ra_real, NoofReal * sizeof(float));
  cudaMalloc((void **)&d_decl_real, NoofReal * sizeof(float));
  cudaMalloc((void **)&d_ra_sim, NoofSim * sizeof(float));
  cudaMalloc((void **)&d_decl_sim, NoofSim * sizeof(float));

  cudaMemcpy(d_ra_real, ra_real, NoofReal * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_decl_real, decl_real, NoofReal * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_ra_sim, ra_sim, NoofSim * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_decl_sim, decl_sim, NoofSim * sizeof(float), cudaMemcpyHostToDevice);

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
