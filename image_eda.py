import os
import PIL
import PIL.Image
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.decomposition import PCA, IncrementalPCA
import time
import glob
import pickle 
from visualization import fashion_scatter
from utils import normalize_data


class ImageEDA:
    """
    This class helps to perform the Exploratory Data Analisys on images
    using a pre-trained model to extract the feature layers and plot 
    the result with a dimensionality reduction algorithm.
    After the fitting and transform the dataset, an output file is 
    generated as a pickle file to further analysis without the need
    of reprocessing the whole dataset.

    Attributes
    ----------
    dataset_name : str
        The name of the dataset being analysed, it is used to generate
        the output file.
    image_path : str
        The full path for the dataset images. They are not read 
        recursively.
    annotations_path : str
        The full path to the csv file containing the boxes and labels 
        in the format (image_path, x, y, w, h, label)
    model : str
        The name of the pre-trained model to be used.
    dr_method : str
        The name of the dimensionality reduction method (PCA, t-SNE).
    batch_size : int
        Number of samples being processes at the sime time for fitting
        in memory.
    n_components : int
        Number of components that the dr_method will be using to the 
        data analysis.
    """

    def __init__(self, dataset_name:str, image_path:str, annotations_path:str = "",
                 model:str = "vgg16", dr_method:str = "pca", batch_size:int = 100, 
                 n_components:int = 2):
        """
        Initialize ImageEDA object based on a pre-existing file for 
        visualization or construct the object for further processing.
        """
        if "pickle" in image_path:
            pickle_path = image_path
            self.load_output(pickle_path)
        else:
            self.image_path = image_path
            self.annotations_path = annotations_path
            self.model_name = model
            self.dr_method = dr_method
            self.batch_size = batch_size
            self.n_components = n_components
            self.dr_object = None
            self.dataset_name = dataset_name
            self.y = None
            self.load_dr_object()
            self.store_sample_labels()
            self.feature_map = None
        self.load_model()
    
    def store_sample_labels(self):
        input_data = pd.read_csv(self.annotations_path)
        self.y = input_data["label"].values
        self.transformed_data = np.empty((self.y.shape[0], self.n_components))

    def __str__(self):
        return f"""
        Dataset: {self.dataset_name}
        Model: {self.model_name}
        DR Method: {self.dr_method}
        Batch size: {self.batch_size}
        N components: {self.n_components}
        """

    def get_input_shape(self):
        """Return the shape of the input data based on the model input"""
        if self.model == None:
            raise Exception("Model not loaded, cannot infer input shape")

        return self.model.layers[0].output.shape[1:]

    def predict_feature_map(self):
        input_data = pd.read_csv(self.annotations_path)
        n_samples = input_data.shape[0]
        self.feature_map = np.empty((n_samples,) + self.model.layers[-1].output.shape[1:])

        for i in range(0, n_samples//self.batch_size):
            # TODO: get dtype from model
            images = np.empty((self.batch_size,) + self.get_input_shape(), dtype=np.int)

            for j, image_path in enumerate(input_data.iloc[i*self.batch_size : (i+1)*self.batch_size]["image_path"]):
                image = PIL.Image.open(os.path.join(self.image_path,  image_path))

                if len(np.array(image).shape) != 3:
                    rgbimg = PIL.Image.new("RGB", image.size)
                    rgbimg.paste(image)
                    image = rgbimg

                image = image.resize( self.get_input_shape()[:-1] )
                image = np.array(image)
                images[j] = image
            self.feature_map[i*self.batch_size : (i+1)*self.batch_size] = self.model(images)

        self.feature_map = normalize_data(self.feature_map)

    def partial_fit(self):
        """
        Fit the dr_method on the data on batches based on the batch_size
        parameter. 
        The data is read based on the annotations_path and image_path 
        attributes. After the fitting, the dr_object parameter will 
        be able to transform the data later.
        """
        if self.dr_method != "pca":
            raise Exception(f"{self.dr_method} does not support batch fit.")

        input_data = pd.read_csv(self.annotations_path)
        n_samples = input_data.shape[0]

        for i in range(0, n_samples//self.batch_size):
            partial_feature_map = self.feature_map[i*self.batch_size : (i+1)*self.batch_size]
            self.dr_object.partial_fit( partial_feature_map )

    def transform(self):
        """
        Transforms the data on batches with the dr_object that should
        already be fitted previously.
        Feed the transformed_data attribute for further visualization.
        """
        input_data = pd.read_csv(self.annotations_path)
        n_samples = input_data.shape[0]

        for i in range(0, n_samples//self.batch_size):
            partial_feature_map = self.feature_map[i*self.batch_size : (i+1)*self.batch_size]
            self.transformed_data[i*self.batch_size : (i+1)*self.batch_size] = self.dr_object.transform( partial_feature_map )

    def load_dr_object(self):
        """Instantiate dr_object based on the selected dr_method"""
        if self.dr_method == "pca":
            self.dr_object = IncrementalPCA(n_components=self.n_components)

    def load_model(self):
        """Load model based on the model_name"""
        if self.model_name == "vgg16":
            model = tf.keras.applications.VGG16(
                include_top=True, weights='imagenet', input_tensor=None, input_shape=None,
                pooling=None, classes=1000, classifier_activation='softmax'
            )
            feature_layer = model.layers[-2].output
            self.model = tf.keras.Model(inputs = model.input, outputs = feature_layer)

    def load_output(self, *args):
        """Load output file based on existing file or object attributes"""
        if len(args) == 1:
            self.load_output_file(args[0])
        else:
            self.load_output_object()

    def load_output_file(self, output_pickle:str):
        """Open the output_pickle file and feed the object"""
        data = pickle.load( open(output_pickle, "rb") )
        self.dataset_name = data["dataset_name"]
        self.model_name = data["model_name"]
        self.dr_method = data["dr_method"]
        self.dr_object = data["dr_object"]
        self.batch_size = data["batch_size"]
        self.n_components = data["n_components"]
        self.transformed_data = data["transformed_data"]
        self.y = data["y"]

    def load_output_object(self):
        """"Open the pickle file and feed the object"""
        data = pickle.load( open(f"{self.dataset_name}_{self.model_name}\
                                 _{self.dr_method}_{self.n_components}.pickle", "rb") )
        self.dr_object = data["dr_object"]
        self.transformed_data = data["transformed_data"]

    def save_output(self):
        """Write the output into a pickle file"""
        with open(f"{self.dataset_name}_{self.model_name}_{self.dr_method}_{self.n_components}.pickle",
                  'wb') as out_file:
            obj = dict()
            obj["dataset_name"] = self.dataset_name
            obj["model_name"] = self.model_name
            obj["dr_method"] = self.dr_method
            obj["dr_object"] = self.dr_object
            obj["batch_size"] = self.batch_size
            obj["n_components"] = self.n_components
            obj["transformed_data"] = self.transformed_data
            obj["y"] = self.y
            pickle.dump(obj, out_file)

    def visualize(self):
        """Plot the transformed_data and show their classes"""
        # TODO: make configurable file with classes and associated ids
        classes = {
            "car": 0,
            "motorbike": 1,
            "truck": 2,
            "bus": 3
        }
        # TODO: extend code to n_components
        pca_df = pd.DataFrame(columns = ['pca1','pca2'])
        pca_df['pca1'] = self.transformed_data[:,0]
        pca_df['pca2'] = self.transformed_data[:,1]
        top_two_comp = pca_df[['pca1','pca2']]
        labels = np.array([classes[x] for x in self.y])

        fashion_scatter(top_two_comp.values, labels, len(classes.keys()))

