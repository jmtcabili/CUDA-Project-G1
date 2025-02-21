# CUDA-Project-G1
Members
1. Johan Marlo T. Cabili
2. Emma Celine Conception R. Cacatian
3. Ashantie Louize B. Demanalata
4. Geo Brian P. Hilomen

## Link to YouTube Video

## Report
### C Output Screenshot
![Screenshot 2025-02-21 191733](https://github.com/user-attachments/assets/3f5cf701-b13a-4f08-8eb7-8458f5b98b42)

### Unified Memory Output Screenshot 
![image](https://github.com/user-attachments/assets/0b123908-c2be-4cc3-b498-a973f9f8eaa5)
![image](https://github.com/user-attachments/assets/81e4e4d0-5211-468a-9727-9aac0672359f)

### Prefetching of Data with Memory Advise Output Screenshot
![image](https://github.com/user-attachments/assets/4ba36b04-50c7-46a5-a3e1-6557e54b1c4f)
![image](https://github.com/user-attachments/assets/f72e555a-987d-4e58-8ccb-8f711d26447a)

### Data Initialization in CUDA Kernel Output Screenshot
![image](https://github.com/user-attachments/assets/dbcd1bfa-cf4d-41de-9bd9-ef87e42b44ed)
![image](https://github.com/user-attachments/assets/0422ee13-fd4a-4b1b-bce4-fd63b8a014c3)

### Old Data Transfer Method Output Screenshot
![image](https://github.com/user-attachments/assets/c6d0c8e7-2c48-4ae4-90e2-4a335b8d9e8d)
![image](https://github.com/user-attachments/assets/89d08e9b-56e4-4980-92de-e6e12d75be55)

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

Disclaimer: 
1. Total time operation was the sum of GPU Activities(averages) + DeviceToHost + HostToDevice + Page Fault (if present) 
2. Calculations can be found at (Appendix A).
3. Some implementations such as the data initialization using cuda & old data transfer method have multiple GPU activities that were included in the total time operation. 
- GPU Activities for data initialization using cuda:  conv1D_kernel(unsigned long, float*, float*), initData(unsigned long, float*)
- GPU Activities for old data transfer method: [CUDA memcpy DtoH], Conv1D(unsigned long, float*, float*), [CUDA memcpy HtoD]

**

**Unified Memory Profile - Page Fault Comparison** 

![image](https://github.com/user-attachments/assets/e9ca2c4f-4baf-4da1-89d3-efde4dabecaf)

Among the four data transfer methods tested, only Unified memory and Data Initialization in the CUDA Kernel resulted in page faults. Unified memory had a longer page fault time than the Data Initialization CUDA Kernel. Page faults occurred on the Unified Memory approach since, at the time the CUDA Kernel attempted to perform the convolution operation, the data had still not been transferred to the device memory. The same can when data was initialized in the CUDA Kernel. Despite both methods already allocating device memory for their data, it is important to note that cudaMallocManaged() does not explicitly store the data yet into the GPU’s memory, so trying to manage variables from the kernel will still result in page faults [https://stackoverflow.com/questions/77624064/getting-gpu-page-fault-by-initializing-data-in-a-kernel]. In the case of the old transfer method, no page fault is observed since cudaMemcpy already transferred the data from host to device, prepping the data to handled by the GPU. 



### Problems Encountered


### Appendices

Appendix A:

<img src="https://github.com/user-attachments/assets/fb11fd26-bc5f-4a66-9b21-f7a4a676f499" alt="AppendixA" width="500"/>

