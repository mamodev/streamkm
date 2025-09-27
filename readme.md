# StreamKM++ HPC Implementation

This project delivers a **high-performance, scalable implementation** of the  [StreamKM++ Paper (ACM Digital Library)](https://dl.acm.org/doi/10.1145/2133803.2184450) streaming *k-means clustering* algorithm, redesigned with modern  
hardware in mind. Unlike the original reference code, our approach is built  
around **cache-aware data layouts** and memory-access patterns, achieving more  
than **15× speedup in the serial version alone**. By keeping the cache  
hierarchy at the core of the design, the algorithm scales linearly both with  
the coreset size and the dimension of the samples.

Beyond the optimized serial implementation, the project provides several  
parallel backends, ranging from shared-memory solutions (**OpenMP**, **native multithreading**, **FastFlow**) to GPU acceleration (**CUDA**, **OpenACC**) and distributed  
execution (**MPI**). This ensures the algorithm can scale seamlessly from  
single-core machines up to large HPC clusters.

The implementation is fully portable, with no operating system dependencies,  
and can be used in batch mode with an **in-memory data stream** or in  
streaming mode with **custom online data streams** such as sockets.


# 🚧⚡ Refactoring in Progress ⚡🚧

**Important Notice**:  
This project is currently under **major refactoring** (with a `git rebase` involved).  
Some parts of the implementation may be temporarily **missing or incomplete**.  

✨ The upcoming update will introduce a **new relaxed version of the algorithm**,  
designed to **break free from strict sequential execution**.  

🔥 This change will **unlock new parallelization strategies**,  
leading to **lower latency** and **even better performance** across all backends.  

⏳ Full, updated implementations will be restored over the **next few weeks**.  

---