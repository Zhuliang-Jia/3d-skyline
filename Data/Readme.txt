This section of the code focuses solely on preparing the dataset for later use in the main comparison. We utilize 3D spatial data from the Open City Model and process it to generate bounding box information and additional non-spatial attributes for each building.

Data Source
The original 3D spatial data comes from the Open City Model project (https://github.com/opencitymodel/opencitymodel)
For our work, we randomly selected some regions from the following states:
-Arkansas
-California
-Florida

Data Processing Steps
1. Using DataSetConvert.ipynb to extract the following bounding box information for each building:
	-max_x, min_x
	-max_y, min_y
	-max_z, min_z
2. We generated 90 additional non-spatial attributes for each building. These attributes are randomly generated and not part of the original dataset.
3. The processed data (bounding boxes + attributes) was saved in CSV format for use in the following tasks.