# CUDA-Project-G1
Members
1. Johan Marlo T. Cabili
2. Emma Celine Conception R. Cacatian
3. Ashantie Louize B. Demanalata
4. Geo Brian P. Hilomen

## Link to YouTube Video
https://youtu.be/rHbWhAE5Jws
## Report
### C Output Screenshot
<img src="https://github.com/user-attachments/assets/3f5cf701-b13a-4f08-8eb7-8458f5b98b42" alt="COutput" width="500"/>

### Unified Memory Output Screenshot 
<img src="https://github.com/user-attachments/assets/0b123908-c2be-4cc3-b498-a973f9f8eaa5" alt="UNF1" width="500"/>
<img src="https://github.com/user-attachments/assets/81e4e4d0-5211-468a-9727-9aac0672359f" alt="UNF2" width="500"/>


### Prefetching of Data with Memory Advise Output Screenshot
<img src="https://github.com/user-attachments/assets/4ba36b04-50c7-46a5-a3e1-6557e54b1c4f" alt="PRF1" width="500"/>
<img src="https://github.com/user-attachments/assets/f72e555a-987d-4e58-8ccb-8f711d26447a" alt="PRF2" width="500"/>

### Data Initialization in CUDA Kernel Output Screenshot
<img src="https://github.com/user-attachments/assets/dbcd1bfa-cf4d-41de-9bd9-ef87e42b44ed" alt="DATIN1" width="500"/>
<img src="https://github.com/user-attachments/assets/0422ee13-fd4a-4b1b-bce4-fd63b8a014c3" alt="DATIN2" width="500"/>

### Old Data Transfer Method Output Screenshot
<img src="https://github.com/user-attachments/assets/c6d0c8e7-2c48-4ae4-90e2-4a335b8d9e8d" alt="OLD1" width="500"/>
<img src="https://github.com/user-attachments/assets/89d08e9b-56e4-4980-92de-e6e12d75be55" alt="OLD2" width="500"/>

All outputs had an error count of 0 and the 1-D Convolution operation was completed in each of the data transfer methods. 

### Execution Table and Analyses
**2^28 Elements Average Run Time**

![image](https://github.com/user-attachments/assets/38469b34-65da-4650-b701-e4e87a2cbf29)

The table shows the average run time of each kernel implementation in CUDA when ran 30 times and then compared against the C kernel.
Observations: 
1. Of the four implementations, prefetching took the shortest amount of time which is only 1% runtime of the C kernel. 
2. The worst performing of the four is the data initialization which is almost half of the C Kernel’s average time.

**2^28 Elements Total Execution Time**

![image](https://github.com/user-attachments/assets/3e329c35-4c3b-493d-8659-9b28023a2ab1)

Observations: 
1. In single kernel run, the unified memory took the longest to execute, taking 1,412 milliseconds to execute. When multiple kernels ran, the unified memory implementation also took the longest. 
2. Of the four implementations, the prefetching with memadvise performed the best with 185.97 ms and 499.44 ms with single kernel and multiple kernel runs, respectively.  
3. On multiple kernel run, an improvement in the minimum time can be observed across all implementations that can be attributed to reduced overhead time when running programs on multiple instances. 

Caveats: 
1. Total time operation was the sum of GPU Activities(averages) + DeviceToHost + HostToDevice + Page Fault (if present) 
2. Calculations can be found at (Appendix A).
3. Some implementations such as the data initialization using cuda & old data transfer method have multiple GPU activities that were included in the total time operation.
    GPU Activities for data initialization using cuda:  conv1D_kernel(unsigned long, float*, float*), initData(unsigned long, float*)
    GPU Activities for old data transfer method: [CUDA memcpy DtoH], Conv1D(unsigned long, float*, float*), [CUDA memcpy HtoD]



**Unified Memory Profile - Page Fault Comparison** 

![image](https://github.com/user-attachments/assets/e9ca2c4f-4baf-4da1-89d3-efde4dabecaf)

Among the four data transfer methods tested, only Unified memory and Data Initialization in the CUDA Kernel resulted in page faults. Unified memory had a longer page fault time than the Data Initialization CUDA Kernel. Page faults occurred on the Unified Memory approach since, at the time the CUDA Kernel attempted to perform the convolution operation, the data had still not been transferred to the device memory. The same can when data was initialized in the CUDA Kernel. Despite both methods already allocating device memory for their data, it is important to note that cudaMallocManaged() does not explicitly store the data yet into the GPU’s memory, so trying to manage variables from the kernel will still result in page faults [https://stackoverflow.com/questions/77624064/getting-gpu-page-fault-by-initializing-data-in-a-kernel]. In the case of the old transfer method, no page fault is observed since cudaMemcpy already transferred the data from host to device, prepping the data to handled by the GPU. 

**Unified Memory Profile - Data Transfer Comparison**
![image](https://github.com/user-attachments/assets/134866d2-56d5-492e-a2cf-bf2f22b1e68d)
Observation:
Old Data Transfer Method took the longest time to transfer data in both Host-to-Device and Device-to-Host. The automatic management and transfer of memory between devices through unified memory performs much better at transferring data as it is 72% faster from host to device and 76% when data moves from device to host. Further improvements are seen with prefetching with mem advise. According to the CUDA programming tutorial by NVIDIA, the reason for the improvement in unified memory lies in the unified and ease of access to data without the need for manual allocation (cudaMalloc) and movement (cudaMemcpy). Especially in the current project, where multiple iterations, unified memory can minimize the distance between the devices that use it often, making it a faster approach to handle how data is migrated between host and device. Then, with initializing data on the CUDA kernel, it performs similar to the plain Unified Memory approach, but since data was initialized in the device kernel, it only has to transfer from device to host. 

https://docs.nvidia.com/cuda/cuda-c-programming-guide/#unified-memory-programming 


**Different quantities of threads - Performance comparison**
![image](https://github.com/user-attachments/assets/056c21fa-ad89-4652-bdb6-eadbc475269b)

Observations: 
1. On the four implementations of the program, the worst performing block size tends to vary per implementation as seen on the table. For example, on the old data transfer method, even though having the maximum number of threads(1024 threads) count performed the fastest, the block with 256 threads performed the worst. Moreover, in prefetching, the block with the maximum number of threads performed the worst. Lastly, although in the unified memory and data initialization the block wiith 32 threads performed the worst, in data initialization, the block with 512 threads performed the second worst.
        
2. Intuitively, a higher thread count should result in a better run time as there are more workers deployed to execute the program. However, in doing this performance comparison under varying quantities of threads, the intuition is proven wrong as this is not the case. Upon research, it was discovered that understanding the constraints of the kernel as well as the GPU it is running on is vital in choosing a block size [https://developer.nvidia.com/blog/cuda-pro-tip-occupancy-api-simplifies-launch-configuration/]. This is where the term occupancy comes in. Occupancy refers to the ratio of active warps on a Streaming Multiprocessor (SM) to the maximum number of active warps that can be supported by the SM[https://docs.nvidia.com/gameworks/content/developertools/desktop/analysis/report/cudaexperiments/kernellevel/achievedoccupancy.htm#:~:text=Occupancy%20is%20defined%20as%20the,be%20different%20for%20each%20SM.]. Moreover, it was affirmed here that having a higher occupancy does not always lead to better performance, hence choosing an appropriate block size is critical.

Formula for calculating performance:
- Old Data Transfer Time = memcpy HtoD + kernel execution (30 loops) + memcpy DtoH
- Unified Memory = execution time + page fault
- Data Initialization through CUDA Kernel = execution time  + page fault 
- Prefetching with Memadvise = execution time + data transfer

### Problems Encountered

### Appendices

Appendix A:

<img src="https://github.com/user-attachments/assets/fb11fd26-bc5f-4a66-9b21-f7a4a676f499" alt="AppendixA" width="500"/>

