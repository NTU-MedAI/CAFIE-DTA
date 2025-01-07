1 System requirements:

	Hardware requirements: 
		Model.py requires a computer with enough RAM to support the in-memory operations.
		Operating system：Linux

	Code dependencies:
		python '3.7' (conda install python==3.7)
		pytorch-GPU '1.10.1' (conda install pytorch==1.12.1 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=10.2 -c pytorch)
		numpy '1.16.5' (conda install numpy==1.21.6)
2 Instructions for use(two benchmark datasets are included in our data):

	Based on kiba dataset:
		First, put folder data_kiba, DataHelper.py, emetrics.py , KIBA_fusion.py ,arg_information_KIBA.py, information.py ,result_process.py, MCAT.py ,GNN.py into the same folder.
		Second, use PyCharm to open KIBA_fusion.py and set the python interpreter of PyCharm.
		Third, modify codes in arg_information_KIBA.py,KIBA_fusion.py, information.py , result_process.py to set the path for loading data and the path for saving the trained model. The details are as follows:
			line 16 in arg_information_KIBA.py
			line 571-575 in KIBA_fusion.py
			line 10 in information.py
			line 73 in result_process.py
		Fourth, open Anaconda Prompt and enter the following command:
			activate env_name
		Fifth, run KIBA_fusion.py in PyCharm.

		Expected output：
			Results (MSE, CI, RM2) predicted by CAFIE-DTA on test set of KIBA dataset for 5 times would be output .

		Expected run time on a "normal" desktop computer:
			The run time in our coumputer (NVIDIA RTX A6000) .

	Based on davis dataset:
		First, put folder data_davis, DataHelper.py, emetrics.py , Davis_fusion.py ,arg_information_Davis.py, information.py ,result_process.py, MCAT.py ,GNN.py into the same folder.
		Second, use PyCharm to open Davis_fusion.py and set the python interpreter of PyCharm.
		Third, modify codes in arg_information_Davis.py,Davis_fusion.py, information.py , result_process.py to set the path for loading data and the path for saving the trained model. The details are as follows:
			line 16 in arg_information_Davis.py
			line 579-583 in Davis_fusion.py
			line 15 in information.py
			line 73 in result_process.py
		Fourth, open Anaconda Prompt and enter the following command:
			activate env_name
		Fifth, run Davis_fusion.py in PyCharm.

		Expected output：
			Results (MSE, CI, RM2) predicted by CAFIE-DTA on test set of davis dataset for 5 times would be output.

		Expected run time on a "normal" desktop computer:
			The run time in our coumputer (NVIDIA RTX A6000) .
