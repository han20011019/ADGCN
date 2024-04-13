# ADGCN
 The project has implemented ADGCN.
Author: Jijie Han
Institution: Beijing Electronic Science and Technology Institute
Contact: 3427525467@qq.com

To run this code, you need PyTorch along with the following dependencies: torch-cluster, torch-geometric, torch-scatter, and torch-sparse.

This code implements the ADGCN method, using the USTC-TFC2016 and ISCX-VPN-NonVPN-2016 datasets.

The USTC-TFC2016 dataset can be obtained from https://github.com/echowei/DeepTraffic.

The ISCX-VPN-NonVPN-2016 dataset can be obtained from https://www.unb.ca/cic/datasets/vpn.html.

Both of these datasets are raw traffic datasets in PCAP format.


If you want to verify whether your runtime environment is properly configured, you can directly run the GCNII.py file, which already includes the required data. This way, you can directly view the running results.

If you want to apply ADGCN to your own data, then follow the steps below.

First, preprocess the raw data using date preprocessing. This tool was developed by Wang et al. and can be obtained from https://github.com/yungshenglu/USTC-TK2016. Refer to the tool documentation for specific usage. In this work, only run 1_Pcap2Session.ps1 and 2_ProcessSession.ps1 to obtain L7+session traffic through these two steps.


Step two, run select_few_shot.py. This code can be used to select 200 data from each class of traffic to construct a small sample dataset.

Step three, use auto_Dense_encoder.py to reconstruct each selected class of traffic.


Step four, use merge_data.py to combine the data. You can choose which types of data to merge for classification.

Step five, madelabel.py can generate a label file to match the categories of the data in the dataset.

Step six, with the label file and data file ready, run calcu_graph.py to construct the graph.

Once everything is set up, you can run GCNII for traffic classification.
