# image-eda
Python package for working with Exploratory Data Analysis on Image data

The image-eda package uses a pre-trained neural network to extract feature maps from a dataset then apply a dimensionality reduction algorithm and finally, plot the transformed data for analysis.

## Usage Guide:

The image-eda package works on 5 steps:

### Creating the ImageEDA object

```python
image_eda = ImageEDA("dataset_name", LocalCsvSource("path/to/annotations.csv", "path/to/images"))
```

### Extract feature maps

```python
image_eda.predict_feature_map()
```

### Fit the dimensionality reduction algorithm

```python
image_eda.partial_fit()
```

### Transform the data


```python
image_eda.transform()
```

### Visualize 

```python
image_eda.visualize()
```
