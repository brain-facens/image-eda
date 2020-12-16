import os
import mlflow
import argparse
from image_eda import ImageEDA

def main():
    
    global args
    parser = argparse.ArgumentParser(description="Main script for ImageEDA")
    parser.add_argument("--dataset_name", type=str, help="Dataset's name")
    parser.add_argument("--data_source", type=str, help="Absolute data source path")
    parser.add_argument("--annotation_source", type=str, help="Absolute data annotation path")
    parser.add_argument("--model_name", type=str, help="Model name. ie: vgg16")
    parser.add_argument("--dr_method", type=str, help="String representing DR method. ie: pca")
    parser.add_argument("--batch_size", type=int, help="Batch size used for training")
    parser.add_argument("--n_components", type=int, help="Components quantity")
    parser.add_argument("--run_name", type=str, help="Experiment run name for mlflow tracking")
    args = parser.parse_args()
    args.dataset_name
    experiment_id = os.getenv("MLFLOW_EXPERIMENT_ID")

    with mlflow.start_run(experiment_id=experiment_id) as curr_run:
        # start_run run_name parameter doesn't works using mlflow cli yet.
        mlflow.set_tag("mlflow.runName", args.run_name)
        os.environ["RUN_ID"] = curr_run.info.run_id

        args.dataset_name.split()
        args.data_source.split()
        args.annotation_source.split()
        '''
         (self, experiment_name:str, data_source:DataSource = None,
                 model:str = "vgg16", dr_method:str = "pca", batch_size:int = 100,
                 n_components:int = 2, pickle_path:str = "")
        '''
        image_eda = ImageEDA(args.experiment_name, args.data_source, args.dr_method, args.batch_size, args.n_components)

        image_eda.predict_feature_map()
        image_eda.partial_fit()
        image_eda.transform()
        image_eda.visualize('classes_config.txt')