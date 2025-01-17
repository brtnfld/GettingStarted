# Compilation w/ Cray Compiler Wrappers
Users are able to build applications on the Polaris login nodes, but may find it convenient to build and test applications on the Polaris compute nodes in short interactive jobs. This also has the benefit of allowing one to quickly submission scripts.
```
$ qsub -I -l select=1,walltime=0:30:00 -A <PROJECT>

$ make -f Makefile.nvhpc clean
$ make -f Makefile.nvhpc

./submit.sh
```
## Example Submission Script
The following submission script will launch 4 MPI ranks on each node allocated. The MPI ranks are bound to CPUs with a depth (stride) of 8, so 1 MPI rank per NUMA. It's important to call out here that each Polaris node only has 4 GPUs, so assigning more MPI ranks to each GPU would likely have a negative impact on performance if NVIDIA's [Multi-Process Service (MPS)](https://docs.nvidia.com/deploy/mps/index.html) is not enabled (see below).
```
#!/bin/sh
#PBS -l select=1:system=polaris
#PBS -l place=scatter
#PBS -l walltime=0:30:00
#PBS -q debug 
#PBS -A <PROJECT>

cd ${PBS_O_WORKDIR}

# MPI example w/ 4 MPI ranks per node spread evenly across cores
NNODES=`wc -l < $PBS_NODEFILE`
NRANKS_PER_NODE=4
NDEPTH=8
NTHREADS=1

NTOTRANKS=$(( NNODES * NRANKS_PER_NODE ))
echo "NUM_OF_NODES= ${NNODES} TOTAL_NUM_RANKS= ${NTOTRANKS} RANKS_PER_NODE= ${NRANKS_PER_NODE} THREADS_PER_RANK= ${NTHREADS}"

mpiexec -n ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} --depth=${NDEPTH} --cpu-bind depth ./hello_affinity
```

### Example Output:
In the default environment, each MPI rank detects all four GPU available on each Polaris node and targets the first visible device. All MPI ranks end up targeting the same GPU in this case, which may not be ideal. The following output was generated during an interactive job on a compute node.
```
./submit.sh 
NUM_OF_NODES= 1 TOTAL_NUM_RANKS= 4 RANKS_PER_NODE= 4 THREADS_PER_RANK= 1
rnk= 0 :  # of devices detected= 4
To affinity and beyond!! nname= x3212c0s25b1n0  rnk= 0  list_cores= (0-7)  num_devices= 4
    [0,0] Platform[ Nvidia ] Type[ GPU ] Device[ NVIDIA A100-SXM4-40GB ]  uuid= GPU-87969d3e-0fb9-9e82-85f7-976f56cd8309
    [0,1] Platform[ Nvidia ] Type[ GPU ] Device[ NVIDIA A100-SXM4-40GB ]  uuid= GPU-df43b0c3-c6be-995a-5102-df8f9a25f7dd
    [0,2] Platform[ Nvidia ] Type[ GPU ] Device[ NVIDIA A100-SXM4-40GB ]  uuid= GPU-d3475abc-8c3d-29da-5ec6-47b48cfcd954
    [0,3] Platform[ Nvidia ] Type[ GPU ] Device[ NVIDIA A100-SXM4-40GB ]  uuid= GPU-46279b47-ba62-bd70-8f44-1adf69bba7c9

To affinity and beyond!! nname= x3212c0s25b1n0  rnk= 1  list_cores= (8-15)  num_devices= 4
    [1,0] Platform[ Nvidia ] Type[ GPU ] Device[ NVIDIA A100-SXM4-40GB ]  uuid= GPU-87969d3e-0fb9-9e82-85f7-976f56cd8309
    [1,1] Platform[ Nvidia ] Type[ GPU ] Device[ NVIDIA A100-SXM4-40GB ]  uuid= GPU-df43b0c3-c6be-995a-5102-df8f9a25f7dd
    [1,2] Platform[ Nvidia ] Type[ GPU ] Device[ NVIDIA A100-SXM4-40GB ]  uuid= GPU-d3475abc-8c3d-29da-5ec6-47b48cfcd954
    [1,3] Platform[ Nvidia ] Type[ GPU ] Device[ NVIDIA A100-SXM4-40GB ]  uuid= GPU-46279b47-ba62-bd70-8f44-1adf69bba7c9

To affinity and beyond!! nname= x3212c0s25b1n0  rnk= 2  list_cores= (16-23)  num_devices= 4
    [2,0] Platform[ Nvidia ] Type[ GPU ] Device[ NVIDIA A100-SXM4-40GB ]  uuid= GPU-87969d3e-0fb9-9e82-85f7-976f56cd8309
    [2,1] Platform[ Nvidia ] Type[ GPU ] Device[ NVIDIA A100-SXM4-40GB ]  uuid= GPU-df43b0c3-c6be-995a-5102-df8f9a25f7dd
    [2,2] Platform[ Nvidia ] Type[ GPU ] Device[ NVIDIA A100-SXM4-40GB ]  uuid= GPU-d3475abc-8c3d-29da-5ec6-47b48cfcd954
    [2,3] Platform[ Nvidia ] Type[ GPU ] Device[ NVIDIA A100-SXM4-40GB ]  uuid= GPU-46279b47-ba62-bd70-8f44-1adf69bba7c9

To affinity and beyond!! nname= x3212c0s25b1n0  rnk= 3  list_cores= (24-31)  num_devices= 4
    [3,0] Platform[ Nvidia ] Type[ GPU ] Device[ NVIDIA A100-SXM4-40GB ]  uuid= GPU-87969d3e-0fb9-9e82-85f7-976f56cd8309
    [3,1] Platform[ Nvidia ] Type[ GPU ] Device[ NVIDIA A100-SXM4-40GB ]  uuid= GPU-df43b0c3-c6be-995a-5102-df8f9a25f7dd
    [3,2] Platform[ Nvidia ] Type[ GPU ] Device[ NVIDIA A100-SXM4-40GB ]  uuid= GPU-d3475abc-8c3d-29da-5ec6-47b48cfcd954
    [3,3] Platform[ Nvidia ] Type[ GPU ] Device[ NVIDIA A100-SXM4-40GB ]  uuid= GPU-46279b47-ba62-bd70-8f44-1adf69bba7c9
```
## GPU Affinity Helper Script
By setting the environment variable `CUDA_VISIBLE_DEVICES` separately for each MPI rank, one can explicitly set which GPUs (and in which order) will be visible. The helper script `set_affinity_gpu_polaris.sh` if provided as an example.
```
$ cat set_affinity_gpu_polaris.sh 

#!/bin/bash
num_gpus=$(nvidia-smi -L | wc -l)
gpu=$((${PMI_LOCAL_RANK} % ${num_gpus}))
export CUDA_VISIBLE_DEVICES=$gpu
echo “RANK= ${PMI_RANK} LOCAL_RANK= ${PMI_LOCAL_RANK} gpu= ${gpu}”
exec "$@"
```
On Polaris, the `num_gpus` variable will be 4. The script can then be inserted in the `mpiexec` command before the application appears as in the following example.
```
mpiexec -n ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} --depth=${NDEPTH} --cpu-bind depth ./set_affinity_gpu_polaris.sh ./hello_affinity
```
### Example Output
After careful inspection of the reported uuids one can confirm that GPUs were uniquely assigned to MPI ranks.
```
NUM_OF_NODES= 1 TOTAL_NUM_RANKS= 4 RANKS_PER_NODE= 4 THREADS_PER_RANK= 1
“RANK= 0 LOCAL_RANK= 0 gpu= 0”
“RANK= 1 LOCAL_RANK= 1 gpu= 1”
“RANK= 2 LOCAL_RANK= 2 gpu= 2”
“RANK= 3 LOCAL_RANK= 3 gpu= 3”
rnk= 0 :  # of devices detected= 1
To affinity and beyond!! nname= x3212c0s25b1n0  rnk= 0  list_cores= (0-7)  num_devices= 1
    [0,0] Platform[ Nvidia ] Type[ GPU ] Device[ NVIDIA A100-SXM4-40GB ]  uuid= GPU-87969d3e-0fb9-9e82-85f7-976f56cd8309

To affinity and beyond!! nname= x3212c0s25b1n0  rnk= 1  list_cores= (8-15)  num_devices= 1
    [1,0] Platform[ Nvidia ] Type[ GPU ] Device[ NVIDIA A100-SXM4-40GB ]  uuid= GPU-df43b0c3-c6be-995a-5102-df8f9a25f7dd

To affinity and beyond!! nname= x3212c0s25b1n0  rnk= 2  list_cores= (16-23)  num_devices= 1
    [2,0] Platform[ Nvidia ] Type[ GPU ] Device[ NVIDIA A100-SXM4-40GB ]  uuid= GPU-d3475abc-8c3d-29da-5ec6-47b48cfcd954

To affinity and beyond!! nname= x3212c0s25b1n0  rnk= 3  list_cores= (24-31)  num_devices= 1
    [3,0] Platform[ Nvidia ] Type[ GPU ] Device[ NVIDIA A100-SXM4-40GB ]  uuid= GPU-46279b47-ba62-bd70-8f44-1adf69bba7c9
```
## MPS Helper Scripts

NVIDIA's [Multi-Process Service (MPS)](https://docs.nvidia.com/deploy/mps/index.html) can help to improve GPU utilization when multiple MPI ranks are utilizing the same GPU. A couple helper scripts (`enable_mps_polaris.sh` and `disable_mps_polaris.sh`) are provided here for convenience. These scripts can be executed using `mpiexec` to start MPS on each node allocated to a job on Polaris. The job submission script `submit_mps.sh` provides an example where a single MPI rank on each node is used to start MPS before launching the main application(s). Additional information on using MPS is available in the ALCF Polaris documentation [here](https://www.alcf.anl.gov/support/user-guides/polaris/queueing-and-running-jobs/example-job-scripts/index.html).

```
#!/bin/sh
#PBS -l select=2:system=polaris
#PBS -l place=scatter
#PBS -l walltime=0:30:00
#PBS -q debug 
#PBS -A <PROJECT>

cd ${PBS_O_WORKDIR}

# MPI example w/ 8 MPI ranks per node spread evenly across cores
NNODES=`wc -l < $PBS_NODEFILE`
NRANKS_PER_NODE=8
NDEPTH=8
NTHREADS=1

NTOTRANKS=$(( NNODES * NRANKS_PER_NODE ))
echo "NUM_OF_NODES= ${NNODES} TOTAL_NUM_RANKS= ${NTOTRANKS} RANKS_PER_NODE= ${NRANKS_PER_NODE} THREADS_PER_RANK= ${NTHREADS}"

# Enable MPS on each node allocated to job
mpiexec -n ${NNODES} --ppn 1 ./enable_mps_polaris.sh

mpiexec -n ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} --depth=${NDEPTH} --cpu-bind depth ./hello_affinity

mpiexec -n ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} --depth=${NDEPTH} --cpu-bind depth ./set_affinity_gpu_polaris.sh ./hello_affinity

# Disable MPS on each node allocated to job
mpiexec -n ${NNODES} --ppn 1 ./disable_mps_polaris.sh

```
## Multiple MPI Applications 

There are four sets of CPUs forming NUMA domains on a Polaris node and each set is closest to one GPU. The topology can be viewed using the `nvidia-smi topo -m` command while on a Polaris compute node.

```
$ nvidia-smi topo -m
GPU0	GPU1	GPU2	GPU3	mlx5_0	mlx5_1 CPU Affinity NUMA Affinity
GPU0	 X 	NV4	NV4	NV4	SYS	SYS	24-31,56-63	3
GPU1	NV4	 X 	NV4	NV4	SYS	PHB	16-23,48-55	2
GPU2	NV4	NV4	 X 	NV4	SYS	SYS	8-15,40-47	1
GPU3	NV4	NV4	NV4	 X 	PHB	SYS	0-7,32-39	0
mlx5_0	SYS	SYS	SYS	PHB	 X 	SYS		
mlx5_1	SYS	PHB	SYS	SYS	SYS	 X 		

Legend:

  X    = Self
  SYS  = Connection traversing PCIe as well as the SMP interconnect between NUMA nodes (e.g., QPI/UPI)
  NODE = Connection traversing PCIe as well as the interconnect between PCIe Host Bridges within a NUMA node
  PHB  = Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)
  PXB  = Connection traversing multiple PCIe bridges (without traversing the PCIe Host Bridge)
  PIX  = Connection traversing at most a single PCIe bridge
  NV#  = Connection traversing a bonded set of # NVLinks
```
Providing `mpiexec` with a list of CPUs and setting `CUDA_VISIBLE_DEVICES` appropriately is one way to launch multiple applications concurrently on a node with each using a distinct set of CPUs and GPUs. The example script `submit_4x8.sh` launches four separate instances of an application each using one NUMA domain and GPU.
```
#!/bin/sh
#PBS -l select=1:system=polaris
#PBS -l place=scatter
#PBS -l walltime=0:30:00
#PBS -q debug 
#PBS -A Catalyst

cd ${PBS_O_WORKDIR}

# MPI example w/ 8 MPI ranks per node spread evenly across cores
NNODES=`wc -l < $PBS_NODEFILE`
NRANKS_PER_NODE=8
NTHREADS=1

nvidia-smi topo -m

NTOTRANKS=$(( NNODES * NRANKS_PER_NODE ))
echo "NUM_OF_NODES= ${NNODES} TOTAL_NUM_RANKS= ${NTOTRANKS} RANKS_PER_NODE= ${NRANKS_PER_NODE} THREADS_PER_RANK= ${NTHREADS}"

export CUDA_VISIBLE_DEVICES=0
mpiexec -n ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} --cpu-bind list:24:25:26:27:28:29:30:31 ./hello_affinity &

export CUDA_VISIBLE_DEVICES=1
mpiexec -n ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} --cpu-bind list:16:17:18:19:20:21:22:23 ./hello_affinity &

export CUDA_VISIBLE_DEVICES=2
mpiexec -n ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} --cpu-bind list:8:9:10:11:12:13:14:15 ./hello_affinity &

export CUDA_VISIBLE_DEVICES=3
mpiexec -n ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} --cpu-bind list:0:1:2:3:4:5:6:7 ./hello_affinity &

wait
```
Note, this example backgrounds each `mpiexec` and waits for them to complete before exiting the script mimicking what one would typically desire in a production scenario. Output from the different executions may be mixed when written to stdout. 