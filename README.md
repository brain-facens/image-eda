# image-eda
Python package for working with Exploratory Data Analysis on Image data.

The image-eda package uses a pre-trained neural network to extract feature maps from a given number of datasets then apply a dimensionality reduction algorithm and finally, plot the transformed data for analysis.

Currently supporting PCA and t-SNE dimension reduction algorithms.

## Usage Guide:

The image-eda package works on the following steps:

### Create the parameters dictionary for the ImageEDA object

```python
data = {'dataset_name': ['dataset_name_01','dataset_name_02','dataset_name_xx'],
        'annot_path': ['path/to/annotations_01.csv','path/to/annotations_02.csv','path/to/annotations_xx.csv'],
        'image_path': ['path/to/images_01','path/to/images_02','path/to/images_xx'],
        'dr_method': ['t-sne','pca','pca']} # t-sne or pca
```

### Then run the ImageEDA class iterating through the dictionary parameters

```python
image_eda = []
for i in range(0,len(data['dataset_name'])):

    image_eda.append(ImageEDA(experiment_name=data['dataset_name'][i],
                              data_source=LocalCsvSource(data['annot_path'][i],data['image_path'][i], data['dataset_name'][i]),
                              dr_method=data['dr_method'][i],
                              batch_size=10))
    print(image_eda[i])

    # Predict
    image_eda[i].predict_feature_map()
    print(data['dataset_name'][i],'predict done')

    # fit
    image_eda[i].partial_fit()
    print(data['dataset_name'][i],'fit done')

    # transform
    image_eda[i].transform()
    print(data['dataset_name'][i],'transform done')
```

### Run the data visualization for each individual dataset

```python
for i in range(0,len(data['dataset_name'])):
    image_eda[i].visualize('classes_config.txt')
```

### Save the eda model output


```python
for i in range(0,len(data['dataset_name'])):
    image_eda[i].save_output()
```

### Load from previously analyzed data 

```python
image_eda = ImageEDA("dataset_name", pickle_path="dataset_name_vgg16_dr_method_.pickle")
image_eda.visualize("classes_config.txt")
```
