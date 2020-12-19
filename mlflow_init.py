import os
import mlflow
import argparse
from image_eda import ImageEDA
from data_sources import LocalCsvSource

def main():
    
    global args
    parser = argparse.ArgumentParser(description="Main script for ImageEDA")
    parser.add_argument("--experiment_name", type=str, help="Experiment's name")
    parser.add_argument("--dataset_name", type=str, help="Dataset's name")
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

    args.dataset_name.split()
    args.image_path.split()    
    args.annotation_path.split()
    args.dr_method.split()

    with mlflow.start_run(experiment_id=experiment_id) as curr_run:
        
        mlflow.set_tag("mlflow.runName", args.run_name)
        os.environ["RUN_ID"] = curr_run.info.run_id
       
        for i in enumerate(args.dataset_name):
            image_eda.append(ImageEDA(experiment_name=args.experiment_name, 
            data_source=LocalCsvSource(args.annotation_path[i],args.image_path[i], args.dataset_name[i]), 
            dr_method=args.dr_method[i], batch_size=args.batch_size, n_components=args.n_components))



        image_eda.predict_feature_map()
        image_eda.partial_fit()
        image_eda.transform()
        image_eda.visualize('classes_config.txt')

if __name__ == "__main__":
    main()