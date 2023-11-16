import os
import multiprocessing

# Start the parallel pool if possible
# Comment this out if failed and change all the parfor instances to for
# try:
#     multiprocessing.set_start_method('spawn')
#     pool = multiprocessing.Pool()
# except:
#     pass

# Add paths
# os.chdir('../Estimation_of_Distribution_Algorithms')
# import setupMatlabEnv_EDA
# setupMatlabEnv_EDA.setup()

# os.chdir('../your_project_directory')  # Replace with your project directory

# Add additional paths
paths = ['./dataProcessing/', './model/', './utils/']
for path in paths:
    os.chdir(path)
    os.environ['PYTHONPATH'] = os.getcwd() + os.pathsep + os.environ.get('PYTHONPATH', '')
    os.chdir('../')

# os.chdir('../Estimation_of_Distribution_Algorithms')
