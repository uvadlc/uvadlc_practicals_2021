# UvA Deep Learning 1 Course - Practicals

This repository contains the code part of the three assignments of the Deep Learning 1 course, Fall 2021 edition. See the respective folders and ans-delft/Canvas for details on the assignment.

## Environment

We provide two conda environments that install all main packages that you will need for the practicals. If there are additional packages needed for specific assignments, we will explicitly note it in the respective READMEs.

The environment `dl2021_cpu.yml` installs all packages for a CPU-only system, while the `dl2021_gpu.yml` installs the versions of packages that support GPUs as well. Please use the GPU version for the Lisa cluster to make use of the GPUs.

## Debugging help

Before posting your question on Piazza, please check our [debugging guide](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/guide3/Debugging_PyTorch.html) on the notebook website and check previous posts on Piazza. Your fellow students will appreciate it if we try to reproduce duplicating questions. If your bug is not solved with the guide, you can ask your TAs by putting your question on Piazza or come to one of our TA sessions.

## FAQ

* __What parts of the code do I have to fill in?__ The parts that are left for you to implement are clearly marked in the code with comments saying `PUT YOUR CODE HERE`. Please put your code between this and the closing comment `END OF YOUR CODE`.
* __How do I know what I have to implement?__ The general task is described in the assignment, and in the individual python files, check the comments above each `PUT YOUR CODE HERE` section carefully.
* __Am I allowed to add code outside of designated places?__ You are allowed to add additional functions outside of the comments if needed. However, in general, this should not be necessary.
* __Am I allowed to create additional python files?__ Yes, you can create additional python files if you want. However, please do not move any blocks into new files that we task you to fill in (i.e. with `PUT YOUR CODE HERE` comments). The code inside those blocks is allowed to refer to functions in other python files though.
* __Am I allowed to change the input or return arguments of a function (e.g. adding another argument)?__ No, please leave the input and return arguments as given except it is explicitly noted in the comments/description of the function. We will run your code through automated tests, and if your code fails due to a change of the argument structure, you might lose points. 
* __Am I allowed to change code that is already provided to us?__ No, we might use this code in the private code checks. Changing existing code can lead to the tests failing and you losing points. You can check with your TA if you really want to change a part of the code.
* __Am I allowed to import packages that are not provided by the default environments?__ In general, try to limit yourself to the necessary packages only, which should be in the provided environment. If there is a package that you think is really necessary for you, ask your TA, and we can check to have it installed when running your code.
* __What are the unittests for?__ In some assignments, we will release unittests. Those will help you to check whether your code has any bugs. Thus, you can use them to debug your code. Note that passing the provided unittests does not necessarily mean that your code has no bugs at all. Not all functions are tested, and not all special cases are checked. Feel free to extend the unittests yourself if you want.
* __Should I submit my pretrained models and downloaded datasets?__ No, please do not submit pretrained models or downloaded datasets unless we explicitly ask for it. It is also in your interest to not keep the dataset in the submission because Canvas can struggle with such large files, and you don't want to have issues last-minute.
* __Can I submit a GitHub link on Canvas instead of a folder?__ No, please submit a zipped version of your code.
* __What folder structure should I submit?__ Please submit only the assignment folder with the structure as given. For example, for assignment 1, submit a zipped folder called `assignment_1_STUDENTID` where you replace your student ID. This folder should contain your filled-in files of the assignment 1 folder of this repository.