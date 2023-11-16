# Initialize the Paths
# initMatEnv()
import initMatEnv
print('(PHASE-1) All Paths set')

# Import the datasets
# take_db()
import dataProcessing.take_db
print('(PHASE-2) Data has been imported')

# Train COBAYN
#generateModels()
import model.generateModels
print('(PHASE-3) COBAYN has been trained')

# Inference Phase --> COBAYN's Predictions
#predictionTableGenerator()
#print('(PHASE-4) COBAYN has finished the prediction process. See it by writing COBAYN')
