import os
import sys
import mlflow
import argparse
from image_eda import ImageEDA
from data_sources import LocalCsvSource

def main():
    
    global args
    parser = argparse.ArgumentParser(description="Main script for ImageEDA")
    parser.add_argument("--experiment_name", type=str, help="Experiment's name")
    parser.add_argument("--dataset_name", type=str, help="Dataset's identification")
    parser.add_argument("--image_path", type=str, help="Absolute image's path")
    parser.add_argument("--annotation_path", type=str, help="Absolute annotation's path")
    parser.add_argument("--model_name", type=str, help="Model name. ie: vgg16")
    parser.add_argument("--dr_method", type=str, help="String representing DR method. ie: pca")
    parser.add_argument("--batch_size", type=int, help="Batch size used for training")
    parser.add_argument("--n_components", type=int, help="Components quantity")
    parser.add_argument("--run_name", type=str, help="Experiment run name for mlflow tracking")
    args = parser.parse_args()
    args.dataset_name
    experiment_id = os.getenv("MLFLOW_EXPERIMENT_ID")

    dataset_name = args.dataset_name.split()
    image_path = args.image_path.split()    
    annotation_path = args.annotation_path.split()
    dr_method = args.dr_method.split()

    if (len(dataset_name) + len(image_path) - len(annotation_path) - len(dr_method)) != 0:
        print('\nError: Incompatible input.\nDataset name, image path, annotation path and dr method arrays MUST have the same length.')
    else:
        with mlflow.start_run(experiment_id=experiment_id) as curr_run:
            
            mlflow.set_tag("mlflow.runName", args.run_name)
            os.environ["RUN_ID"] = curr_run.info.run_id

            image_eda = []

            for i in range(len(dataset_name)):
                image_eda.append(ImageEDA(experiment_name=args.experiment_name, 
                data_source=LocalCsvSource(annotation_path[i],image_path[i], dataset_name[i]), 
                dr_method=dr_method[i], batch_size=args.batch_size, n_components=args.n_components))
                image_eda[i].predict_feature_map()
                image_eda[i].partial_fit()
                image_eda[i].transform()
                image_eda[i].save_output()
                image_eda[i].mlflow_log('classes_config.txt')

if __name__ == "__main__":
    main()