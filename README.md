# I-DNAN6mA
I-DNAN6mA: Accurate identification of DNA N6-methyladenine sites using base-pairing map and deep learning

## Pre-requisite:  
   - Linux system
   - python3.7
   - pytorch [pytorch version 1.3.1](https://pytorch.org/)
   - PyFeat [PyFeat](https://github.com/mrzResearchArena/PyFeat/)
    

## Installation:

*Install and configure the softwares of python3.7, Pytorch, Infernal, and PyFeat in your Linux system. Please make sure that python3 includes the modules of 'os', 'Sklearn', 'math', 'numpy', 'configparser','random', and 'sys'. If any one modules does not exist, please using 'pip install xxx' command install the python revelant module. Here, "xxx" is one module name.

*Download this repository at https://github.com/XueQiangFan/I-DNAN6mA (xxMB). Then, uncompress it and run the following command lines on Linux System.

~~~
  $ jar xvf I-DNAN6mA-main.zip
  $ chmod -R 777 ./I-DNAN6mA-main.zip
  $ cd ./I-DNAN6mA-main
  $ java -jar ./Util/FileUnion.jar ./save_model/ ./save_model.zip
  $ rm -rf ./save_model
  $ unzip save_model.zip 
  $ cd ../
~~~
Here, you will see one configuration files.   
*Configure the following tools or databases in I-DNAN6mA.config  
  The file of "I-DNAN6mA.config" should be set as follows:
- test_path
~~~
  For example:  
  [test_path]
  test_path = /DNAN6mAsites/dataset/A.thaliana_test.xlsx
~~~

## Run I-DNAN6mA 
### run: python main.py -test_path -o result path
~~~
    For example:
    python main.py -test_path /DNAN6mAsites/dataset/A.thaliana_test.xlsx -result_path ./result
~~~

## The 6mA sites result

*The 6mA sites results of each DNA sequence should be found in the outputted file. In each result file, where "NO" is the position of each residue in your RNA, where "AA" is the name of each residue in your RNA, where "RSA" is the predicted relative accessible surface area of each residue in your RNA, and where "ASA" is the predicted accessible surface area of each nucleotide in your RNA.

## Update History:

First release 2021-08-25

## References

[1]  Xue-Qiang Fan, xxx, Jun Hu, Dong-Jun Yu, and Zhong-Yi Guo*. I-DNAN6mA: Accurate identification of DNA N6-methyladenine sites using base-pairing map and deep learning. sumitted.
