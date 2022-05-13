I-DNAN6mA: *Accurate identification of DNA N6-methyladenine sites using base-pairing map and deep learning.*
====

Contents
----
  * [Abstract](#abstract)
  * [System Requirments](#system-requirments)
  * [Installation](#installation)
  * [Usage](#usage)
  * [Datasets](#datasets)
  * [Citation guide](#citation-guide)
  * [Licence](#licence)
  * [Contact](#contact)


Abstract
----
Motivation: The recent discovery of numerous DNA N6-methyladenine (6mA) has transformed our perception about the roles of 6mA in living organisms. However, our ability to understand them is hampered by our inability to identify 6mA sites rapidly and cost-efficiently by existing experimental methods. Developing a novel method to fast and accurately identify 6mA sites is critical for speeding up the progress of its function detection and understanding.
Results: We propose a novel computational method, I-DNAN6mA, to identify 6mA sites and well complement experimental methods, by leveraging the base-pairing rules and a well-designed three-stage deep learning model with pairwise inputs. The performance of our proposed method is benchmarked and evaluated on four species, i.e., Arabidopsis thaliana, Drosophila melanogaster, Rice, and Rosaceae. The experimental results demonstrate that the I-DNAN6mA achieves accuracies of 91.5%, 92.7%, 88.2%, and 96.2%, Mathew’s correlation coefficient values of 0.855, 0.831, 0.763, and 0.924, and area under the receiver operating characteristic curve values of 0.967, 0.963, 0.947, and 0.990 on four benchmark datasets, respectively, and outperforms several existing state-of-the-art methods. To our knowledge, I-DNAN6mA is the first approach to identify 6mA sites using a novel image-like representation of DNA sequences and a deep learning model with pairwise inputs. I-DNAN6mA is expected to be useful for locating functional regions of DNA. 

System Requirments
----

**Hardware Requirments:**
I-DNAN6mA requires only a standard computer with around 32 GB RAM to support the in-memory operations.

**Software Requirments:**
* [Python3.7](https://docs.python-guide.org/starting/install3/linux/)
* [Pytorch](https://pytorch.org/)
* [Anaconda](https://anaconda.org/anaconda/virtualenv)
* [CUDA 10.0](https://developer.nvidia.com/cuda-10.0-download-archive) (Optional If using GPU)
* [cuDNN (>= 7.4.1)](https://developer.nvidia.com/cudnn) (Optional If using GPU)

I-DNAN6mA has been tested on Ubuntu 18.04 and Window10 operating systems

Installation
----

To install I-DNAN6mA and it's dependencies following commands can be used in terminal:

1. `git clone https://github.com/XueQiangFan/I-DNAN6mA.git`
2. `cd I-DNAN6mA`

Either follow **virtualenv** column steps or **conda** column steps to create virtual environment and to install I-DNAN6mA dependencies given in table below:<br />

|  | &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; conda |
| :- | :--- |
| 3. |  `conda create -n venv python=3.7` |
| 4. |  `conda activate venv` | 
| 5. |  *To run I-DNAN6mA on CPU:*<br /> <br /> `conda install pytorch torchvision torchaudio cpuonly -c pytorch` <br /> <br /> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; *or* <br /> <br />*To run I-DNAN6mA on GPU:*<br /> <br /> `conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch` |
| 6. | `while read p; do conda install --yes $p; done < requirements.txt` | 

Usage
----

**To run the I-DNAN6mA**
### run: python main.py -test_path test path -result_path result path
~~~
    For example:
    python main.py -test_path /DNAN6mAsites/dataset/A.thaliana_test.xlsx -result_path ./result.csv
~~~


Datasets
----

The following dataset was used for Training, Validation, and Testing of RNAsnap2:
[Dropbox](https://github.com/XueQiangFan/I-DNAN6mA/tree/main/Benchmark%20datasets)

Citation guide
----

**If you use I-DNAN6mA for your research please cite the following papers:**

[1]  Xue-Qiang Fan, xxx, Jun Hu, Dong-Jun Yu, and Zhong-Yi Guo*. I-DNAN6mA: Accurate identification of DNA N6-methyladenine sites using base-pairing map and deep learning. sumitted.

Licence
----
Mozilla Public License 2.0

Contact
----
xstrongf@163.com
