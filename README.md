# Handwritten Symbols Classification 
This Repo contains our scripts for the [EEL 5840: Fundamentals of Machine Learning] Fall 2021 Course.

## How to run
- Clone `https://github.com/foundationsofmachinelearning-fa21/project-squid_gamers-ml.git`
- Run `train.py`. Change parameters according to your preferences from the train.py file before training. The changeable parameters are listed below:
- Changeable parameters:
    ```
    Directory or Path
    SEED = Seed value
    batch_size
    learning_rate 
    patience
    epochs
    weight decays
    Augmentation probabilities and rotation angles
    Train Set  
    ```
- For the `test.py` you have to set your directory on the following snippet:
    ```
    model_url = "https://dl.dropboxusercontent.com/s/4jm7u6tw9gdvkh3/resnet18_50-Ag2_epochs_saved_weights.pth?dl=0" #Do NOT change this
    print('Downloading file...')
    urllib.request.urlretrieve(model_url, 'C:\\Users\\HP\\Documents\\MLProject\\resnet18_50-Ag2_epochs_saved_weights.pth') #Set Directory here
    print('Download completed')
    ```
- Do NOT change anything from the above snippet other than the `C:\\Users\\HP\\Documents\\MLProject\\` part [Line 24]. Set the directory exactly where you would clone the repo. For example all my files (train.py, test.py etc.) were in C:>Users>HP>Documents>MLProject. Do NOT change the name of the `.pth` file.
- For the `test.py` file just write the name of the test file inside the function parenthesis. You do not need to add `.npy`.
  For example, I have a test images set as `test_images.npy` and my function is `Test_func()`. To execute run ``Test_func('test_images')``[Line: 158].
  Similarly, for `evaluation()` function I added, if the labels set is `test_labels.npy` run ``evaluation('test_labels')``[Line: 175].
  Note that the `.npy` files should be on the same directory as the `test.py` files, so should be the `weight(.pth)` file provided. If not, change the directory as required.
  The `Test_func()` function takes test images as input and gives an output of predicted classes as numpy array. 
  The `evaluation()` function takes true test labels as input and gives accuracy score, classification report and an image of confusion matrix as output.
  I have attached a sample test set `Images_Sample_Test.npy` & `Labels_Sample_Test.npy` and used them in the functions explained above. You can look how they were loaded to get a neat idea.

## Libraries Used
- The following libraries must be installed in the environment in which the code is being run
- Required Libraries:
    ```
    Numpy
    Torch
    Torchvision
    Opencv-python (cv2) 
    Pandas
    Matplotlib
    Seaborn
    Scikit Learn
    Urllib
    Time [For train.py only]
    Copy [For train.py only]
    ```