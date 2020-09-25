import os
import PIL
import PIL.Image
import numpy as np
from utils import crop_box

class DataSource:

    def count(self):
        """
        Returns the amount of rows
        """
        return 0

    def get_column_values(self, column_name):
        return []

    def batch_process(self, batch_size, process):
        """
        Processes by dividing the data into batches.
        The process function should accept a list as a parameter.
        """
        raise Exception("Not implemented")

    def process_image(self, image, crop_x, crop_y, crop_w, crop_h, size):
        # Crops it
        image = crop_box(image, crop_x, crop_y, crop_w, crop_h)

        # Rescales it
        image = image.resize(size)
        
        # Converts it to RGB
        image = image.convert('RGB')

        # Converts it into a numpy array
        return np.array(image)

    def load_image(self, file_path, crop_x, crop_y, crop_w, crop_h, size):
        """
        Loads the image file, crops and resizes it to the specified parameters
        """
        raise Exception("Not implemented")

        

class LocalCsvSource(DataSource):

    def __init__(self, file_path, image_path):
        import pandas as pd
        
        self.file_path = file_path if isinstance(file_path, list) else [file_path]
        self.image_path = image_path if isinstance(image_path, list) else [image_path]

        
        for file in self.file_path:
            self.data = self.data.append(pd.read_csv(file))
        
        self.data["dataset_name"] = "dataset_name"
        #self.data = [pd.read_csv(self.file_path) for path in self.file_path]

    def count(self):
        return self.data.shape[0]

    def get_column_values(self, column_name):
        return self.data["label"].values

    def batch_process(self, batch_size, process):
        for i in range(0, self.count()//batch_size):
            process(i, self.data.iloc[i*batch_size : (i+1)*batch_size].itertuples())

    def load_image(self, file_path, crop_x, crop_y, crop_w, crop_h, size):
        # Loads the image file
        image = PIL.Image.open(os.path.join(self.image_path, file_path))

        # Crops and resizes the image
        return self.process_image(image, crop_x, crop_y, crop_w, crop_h, size)

class SparkSource(DataSource):
    import pyspark.sql as spark

    def __init__(self, spark_session: spark.SparkSession, file_path, image_path):
        self.file_path = file_path
        self.image_path = image_path
        self.data = spark_session.read.format("csv").option("header", "true").load(self.file_path)

    def count(self):
        return self.data.count()

    def get_column_values(self, column_name):
        return np.array(self.data.select(column_name).rdd.flatMap(lambda x : x).collect())

    def __partition(self, chain, batch_size):
        """
        Partitions an iterator into lists of batch_size
        """
        going = True
        while going:
            data = []
            for _ in range(batch_size):
                try:
                    data.append(next(chain))
                except StopIteration:
                    going = False

            if len(data) > 0:
                yield data

    def batch_process(self, batch_size, process):
        # Splits each partition, so it can group the batches per partition
        # Groups the batches by the specified size
        # Runs the process function with the whole batch as an in-memory list
        self.data.foreachPartition(lambda chain : (process(i, batch) for i, batch in enumerate(self.__partition(chain, batch_size))))

    def load_image(self, file_path, crop_x, crop_y, crop_w, crop_h, size):
        # Loads the image file
        image = PIL.Image.open(os.path.join(self.image_path, file_path))

        # Crops and resizes the image
        return self.process_image(image, crop_x, crop_y, crop_w, crop_h, size)
