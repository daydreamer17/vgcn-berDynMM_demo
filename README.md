# The graph
-*VocabularyGraph.py*:the vocabulary-level graph we build
-*CharacterGraph.py*:the character-level graph we build

# The model
-*vgcn-bert.py*:The proposed model we use
-*vgcn-bertNoChar.py*:The model that move away CharCNN
-*vgcn-bertNoDynMM.py*:The model that move away DynMM

# The dataset
-*custom_dataset.py*:To split the dataset into validate and train and pre-process the data

# To run the code
-*MultipleFusion.py*:The code that train the proposed model and get the metrics in 1,6,12 layers
-*NoFusion.py*:The code that train the vgcn-bertNoChar and get the metrics in 1,6,12 layers
-*FusionWithoutDynMM*:The code that train the vgcn-bertNoDynMM and get the metrics in 1,6,12 layers
