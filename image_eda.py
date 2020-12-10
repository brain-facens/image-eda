import os
import PIL
import PIL.Image
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.manifold import TSNE
import time
import glob
import pickle
import mlflow
import argparse
from visualization import fashion_scatter, plot_components
from utils import normalize_data, crop_box
from data_sources import DataSource

class ImageEDA:
    """
    This class helps to perform the Exploratory Data Analisys on images
    using a pre-trained model to extract the feature layers and plot 
    the result with a dimensionality reduction algorithm.
    After fitting and transforming the dataset, an output file is 
    generated as a pickle file to further analysis without the need
    of reprocessing the whole dataset.

    Attributes
    ----------
    experiment_name : str
        The name of the dataset being analysed, it is used to generate
        the output file.
    data_source : DataSource
        The DataSource instance with the path to the dataset images and
        the csv file containing the boxes and labels in the format
        (image_path, x, y, w, h, label)
    model : str
        The name of the pre-trained model to be used.
    dr_method : str
        The name of the dimensionality reduction method (PCA, t-SNE).
    batch_size : int
        Number of samples being processed at the sime time for fitting
        in memory.
    n_components : int
        Number of components that the dr_method will be using to the 
        data analysis.
    pickle_path : str
        The path for the pickle output file
    """

    def __init__(self, experiment_name:str, data_source:DataSource = None,
                 model:str = "vgg16", dr_method:str = "pca", batch_size:int = 100,
                 n_components:int = 2, pickle_path:str = ""):
        """
        Initialize ImageEDA object based on a pre-existing file for 
        visualization or construct the object for further processing.
        """
        if pickle_path:
            self.load_output(pickle_path)
        else:
            self.data_source = data_source
            self.model_name = model
            self.dr_method = dr_method
            self.batch_size = batch_size
            self.n_components = n_components
            self.dr_object = None
            self.experiment_name = experiment_name
            self.y = None
            self.load_dr_object()
            self.store_sample_labels()
            self.feature_map = None
        self.load_model()
    
    def store_sample_labels(self):
        self.y = self.data_source.get_column_values("label")
        self.transformed_data = np.empty((self.y.shape[0], self.n_components))

    def __str__(self):
        return f"""
        Dataset: {self.experiment_name}
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
        """Pass the dataset through the selected model and store the output"""
        n_samples = self.data_source.count()
        self.feature_map = np.empty((n_samples,) + self.model.layers[-1].output.shape[1:])

        input_shape = self.get_input_shape()
        size = input_shape[:-1]
        images_shape = (self.batch_size,) + input_shape

        def process_batch(i, batch):
            # TODO: get dtype from model
            images = np.empty(images_shape, dtype=np.int)

            for j, data in enumerate(batch):
                images[j] = self.data_source.load_image(data.image_path, data.x, data.y, data.w, data.h, size)

            self.feature_map[i*self.batch_size : (i+1)*self.batch_size] = self.model(images)

        self.data_source.batch_process(self.batch_size, process_batch)
        self.feature_map = normalize_data(self.dr_object, self.feature_map)

    def partial_fit(self):
        """
        Fit the dr_method on the data on batches based on the batch_size
        parameter. 
        The data is read based on the data source annotation attributes.
        After the fitting, the dr_object parameter will be able to
        transform the data later.
        """

        n_samples = self.data_source.count()

        if self.dr_method != "pca":
            '''for i in range(0, n_samples//self.batch_size):
                partial_feature_map = self.feature_map[i*self.batch_size : (i+1)*self.batch_size]
                self.transformed_data[i*self.batch_size : (i+1)*self.batch_size] = self.dr_object.fit_transform(partial_feature_map)'''

            self.transformed_data = self.dr_object.fit_transform(self.feature_map)

        else:
            for i in range(0, n_samples//self.batch_size):
                partial_feature_map = self.feature_map[i*self.batch_size : (i+1)*self.batch_size]
                self.dr_object.partial_fit(partial_feature_map)

    def transform(self):
        """
        Transforms the data on batches with the dr_object that should
        already be fitted previously.
        Feed the transformed_data attribute for further visualization.
        """

        if self.dr_method == 'pca':

            n_samples = self.data_source.count()

            for i in range(0, n_samples//self.batch_size):
                partial_feature_map = self.feature_map[i*self.batch_size : (i+1)*self.batch_size]
                self.transformed_data[i*self.batch_size : (i+1)*self.batch_size] = self.dr_object.transform(partial_feature_map)

        else:
            pass

    def load_dr_object(self):
        """Instantiate dr_object based on the selected dr_method"""
        if self.dr_method == "pca":
            self.dr_object = IncrementalPCA(n_components=self.n_components)
        else:
            self.dr_object = TSNE(n_components=self.n_components, perplexity=100, n_iter=5000, learning_rate=200)

    def load_model(self):
        """Load model based on the model_name"""
        if self.model_name == "vgg16":
            model = tf.keras.applications.VGG16(
                include_top=True, weights='imagenet', input_tensor=None, input_shape=None,
                pooling=None, classes=1000, classifier_activation='softmax'
            )
            
        elif self.model_name == "Xception":
            model = tf.keras.applications.Xception(
                include_top=True,
                weights="imagenet",
                input_tensor=None,
                input_shape=None,
                pooling=None,
                classes=1000,
                classifier_activation="softmax",
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
        with open(output_pickle, "rb") as output_file:
            data = pickle.load(output_file)
        self.experiment_name = data["experiment_name"]
        self.model_name = data["model_name"]
        self.dr_method = data["dr_method"]
        self.dr_object = data["dr_object"]
        self.batch_size = data["batch_size"]
        self.n_components = data["n_components"]
        self.transformed_data = data["transformed_data"]
        self.y = data["y"]

    def load_output_object(self):
        """"Open the pickle file and feed the object"""
        with open(open(f"{self.experiment_name}_{self.model_name}\
                        _{self.dr_method}_{self.n_components}.pickle", 
                        "rb")) as output_file:
            data = pickle.load(output_file)
        self.dr_object = data["dr_object"]
        self.transformed_data = data["transformed_data"]

    def save_output(self):
        """Write the output into a pickle file"""
        with open(f"{self.experiment_name}_{self.model_name}_{self.dr_method}_{self.n_components}.pickle",
                  'wb') as out_file:
            obj = dict()
            obj["experiment_name"] = self.experiment_name
            obj["model_name"] = self.model_name
            obj["dr_method"] = self.dr_method
            obj["dr_object"] = self.dr_object
            obj["batch_size"] = self.batch_size
            obj["n_components"] = self.n_components
            obj["transformed_data"] = self.transformed_data
            obj["y"] = self.y
            pickle.dump(obj, out_file)

    def visualize(self, file):
        """Plot the transformed_data and show their classes"""
        # TODO: make configurable file with classes and associated ids
        classes = {}

        classes_file = open(file)
        
        for index, line in enumerate(classes_file.readlines()):
            classes[line.rstrip('\n')] = index

        classes_file.close()

        # TODO: extend code to n_components
        pca_df = pd.DataFrame(columns = ['pca1','pca2'])
        pca_df['pca1'] = self.transformed_data[:,0]
        pca_df['pca2'] = self.transformed_data[:,1]
        top_two_comp = pca_df[['pca1','pca2']]
        labels = np.array([classes[x] for x in self.y])

        fashion_scatter(top_two_comp.values, labels, len(classes.keys()))

    def visualize_components(self, n_components=10):
        """Plot number of components vs cummulative variance"""
        pca = PCA().fit(self.feature_map)
        plot_components(pca, n_components)
def main():
    
    global args
    parser = argparse.ArgumentParser(description="Main script for training splice-smartcam nn")
    parser.add_argument("--dataset_name", type=str, help="Dataset's name")
    parser.add_argument("--data_source", type=str, help="Absolute data source path")
    parser.add_argument("--annotation_source", type=str, help="Absolute data annotation path")
    parser.add_argument("--model_name", type=str, help="Model name. ie: vgg16")
    parser.add_argument("--dr_method", type=str, help="String representing DR method. ie: pca")
    parser.add_argument("--batch_size", type=int, help="Batch size used for training")
    parser.add_argument("--n_components", type=int, help="Components quantity")
    parser.add_argument("--run_name", type=str, help="Experiment run name for mlflow tracking")
    args = parser.parse_args()
    mlflow.tensorflow.autolog()
    experiment_id = os.getenv("MLFLOW_EXPERIMENT_ID")

    with mlflow.start_run(experiment_id=experiment_id) as curr_run:
        # start_run run_name parameter doesn't works using mlflow cli yet.
        mlflow.set_tag("mlflow.runName", args.run_name)
        os.environ["RUN_ID"] = curr_run.info.run_id

        # Start training workflow
        '''
        ?
        '''
        tf.compat.v1.app.run(main=training)
if __name__ == "__main__":
    main()
