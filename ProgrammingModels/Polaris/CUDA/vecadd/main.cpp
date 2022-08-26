#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <cassert>
#include <hdf5.h>
#include <cstring>

#define _N 1024
#define _LOCAL_SIZE 64

#ifdef _SINGLE_PRECISION
  typedef float real_t;
#else
  typedef double real_t;
#endif

#define _CUDA_CHECK_ERRORS()               \
{                                          \
  cudaError err = cudaGetLastError();	   \
  if(err != cudaSuccess) {		   \
    std::cout				   \
      << "CUDA error with code "           \
      << cudaGetErrorString(err)	   \
      << " in file " << __FILE__           \
      << " at line " << __LINE__	   \
      << ". Exiting...\n";		   \
    exit(1);				   \
  }                                        \
}

// ----------------------------------------------------------------

// kernel to offload
  
__global__ void _vecadd(real_t * a, real_t * b, real_t * c, int n)
{
  int id = blockIdx.x * blockDim.x + threadIdx.x;

  if(id < n) c[id] = a[id] + b[id];
}
 
// ----------------------------------------------------------------
						 
int main( int argc, char* argv[] )
{
  // Allocate memory on host

  const int N = _N;
  
  real_t * a = (real_t*) malloc(N*sizeof(real_t));
  real_t * b = (real_t*) malloc(N*sizeof(real_t));
  real_t * c = (real_t*) malloc(N*sizeof(real_t));
  
  // Initialize host
  for(int i=0; i<N; ++i) {
    a[i] = sin(i)*sin(i);
    b[i] = cos(i)*cos(i);
    c[i] = -1.0;
  }
  
  // Number of total work items
  
  size_t grid_size = (int)ceil((_N)/_LOCAL_SIZE);
  size_t block_size = _LOCAL_SIZE;
  
  // Cuda Properties
  int num_devices;
  cudaGetDeviceCount(&num_devices);
  _CUDA_CHECK_ERRORS();
  printf("# of devices= %i\n",num_devices);
  
  for(int i=0; i<num_devices; ++i) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    _CUDA_CHECK_ERRORS();
    
    char name[256];
    strcpy(name, prop.name);
    
    printf("  [%i] Platform[ Nvidia ] Type[ GPU ] Device[ %s ]\n", i, name);
  }  
  
  // Device ID
  
  int device_id = 0;
  printf("Running on GPU %i!\n",device_id);
  
#ifdef _SINGLE_PRECISION
  printf("Using single-precision\n\n");
#else
  printf("Using double-precision\n\n");
#endif
  
  // check cuda version
  
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, device_id);
  _CUDA_CHECK_ERRORS();
  
  printf("  Name= %s\n",prop.name);
  printf("  Locally unique identifier= %s\n",prop.luid);
  
  printf("  Clock Frequency(KHz)= %i\n",prop.clockRate);
  printf("  Compute Mode= %i\n",prop.computeMode);
  printf("  Major compute capability= %i\n",prop.major);
  printf("  Minor compute capability= %i\n",prop.minor);
  
  printf("  Number of multiprocessors on device= %i\n",prop.multiProcessorCount);
  printf("  Warp size in threads= %i\n",prop.warpSize);
  printf("  Single precision performance ratio= %i\n",prop.singleToDoublePrecisionPerfRatio);
  
  // Create device buffers
  
  real_t * d_a;
  real_t * d_b;
  real_t * d_c;

  int size = N * sizeof(real_t);

  cudaMalloc(&d_a, size);
  cudaMalloc(&d_b, size);
  cudaMalloc(&d_c, size);
  
  // Transfer data to device

  cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);  _CUDA_CHECK_ERRORS();
  cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);  _CUDA_CHECK_ERRORS();
  cudaMemcpy(d_c, c, size, cudaMemcpyHostToDevice);  _CUDA_CHECK_ERRORS();
  
  // Execute kernel

  _vecadd<<<grid_size, block_size>>>(d_a, d_b, d_c, _N);
  _CUDA_CHECK_ERRORS();
  
  // Transfer data from device

  cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost); _CUDA_CHECK_ERRORS();
  
  //Check result on host
  
  double diff = 0;
  for(int i=0; i<N; ++i) diff += (double) c[i];
  diff = diff/(double) N - 1.0;
  
  if(diff*diff < 1e-6) printf("\nResult is CORRECT!! :)\n");
  else printf("\nResult is WRONG!! :(\n");
  
  // Clean up

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  
  free(a);
  free(b);

  // IO with HDF5

  int mpi_size, mpi_rank;
  MPI_Comm comm  = MPI_COMM_WORLD;
  MPI_Info info  = MPI_INFO_NULL;
  hid_t       file_id, dset_id;         /* file and dataset identifiers */
  hid_t       filespace, memspace;      /* file and memory dataspace identifiers */
  hsize_t     dimsf[1];                 /* dataset dimensions */
  hsize_t     count[1];	          /* hyperslab selection parameters */
  hsize_t     offset[1];
  hid_t	      plist_id;                 /* property list identifier */
  int         i;
  //
  // Initialize MPI
  //
  MPI_Init(&argc, &argv);
  MPI_Comm_size(comm, &mpi_size);
  MPI_Comm_rank(comm, &mpi_rank);

  dimsf[0] = N;

  if(mpi_rank == 0) {
    //
    // Set up file access property list with parallel I/O access
    //

    plist_id = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_libver_bounds(plist_id, H5F_LIBVER_LATEST, H5F_LIBVER_LATEST);
    hid_t dcpl_id = H5Pcreate(H5P_DATASET_CREATE);
    H5Pset_alloc_time(dcpl_id, H5D_ALLOC_TIME_EARLY);
    H5Pset_fill_time(dcpl_id, H5D_FILL_TIME_NEVER);

    //
    // Create a new file collectively and release property list identifier.
    //

    file_id = H5Fcreate("vadd.h5", H5F_ACC_TRUNC, H5P_DEFAULT, plist_id);
    H5Pclose(plist_id);
        
    //
    // Create the dataspace for the dataset.
    //
    filespace = H5Screate_simple(1, dimsf, NULL);
        
    dset_id = H5Dcreate(file_id, "DSET", H5T_NATIVE_FLOAT, filespace, H5P_DEFAULT, dcpl_id, H5P_DEFAULT);
    H5Dclose(dset_id);

    H5Sclose(filespace);
    H5Fclose(file_id);
    H5Pclose(dcpl_id);
  }

  plist_id = H5Pcreate(H5P_FILE_ACCESS);

  H5Pset_coll_metadata_write(plist_id, 1);
  H5Pset_all_coll_metadata_ops(plist_id, 1 );
  H5Pset_libver_bounds(plist_id, H5F_LIBVER_LATEST, H5F_LIBVER_LATEST);

  H5Pset_fapl_mpio(plist_id, comm, info);
      
  //
  // Create a new file collectively and release property list identifier.
  //
  file_id = H5Fopen("vadd.h5", H5F_ACC_RDWR, plist_id);
  H5Pclose(plist_id);
      
  //
  // Create the dataspace for the dataset.
  //
  filespace = H5Screate_simple(1, dimsf, NULL);

  //
  // Each process defines dataset in memory and writes it to the hyperslab
  // in the file.
  //
  count[0] =  dimsf[0]/mpi_size; 
  offset[0] = (mpi_rank)*count[0];
  memspace = H5Screate_simple(1, count, NULL);

  //
  // Select hyperslab in the file.
  //

  H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offset, NULL, count, NULL);
      
  plist_id = H5Pcreate(H5P_DATASET_XFER);

  //  H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);

  dset_id = H5Dopen(file_id, "DSET", H5P_DEFAULT);
  H5Dwrite(dset_id, H5T_NATIVE_FLOAT, memspace, filespace, plist_id, c);
  H5Dclose(dset_id);

  //
  // Close/release resources.
  //
  H5Pclose(plist_id);
  H5Sclose(memspace);
  H5Sclose(filespace);
  H5Fclose(file_id);

  free(c);
}
