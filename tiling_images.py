from pathml.core import SlideData, types
from pathml.preprocessing import Pipeline, NucleusDetectionHE, StainNormalizationHE, TissueDetectionHE # pipeline example
from dask.distributed import Client, LocalCluster # NA
from pathml.ml import TileDataset
from torch.utils.data import DataLoader


# Set up a local cluster for distributed computing (optional)
cluster = LocalCluster(n_workers=6) # NA
client = Client(cluster) # NA

# Load the image and assign the image type
wsi = SlideData("./data/CMU-1.svs", name = "example", slide_type = types.HE)

# Create a pipeline to process the image
pipeline = Pipeline([
    TissueDetectionHE(mask_name = "tissue", min_region_size=500,
                      threshold=30, outer_contours_only=True),
    StainNormalizationHE(target="normalize", stain_estimation_method="macenko"),
    NucleusDetectionHE(mask_name = "detect_nuclei")
])

# Run the pipeline on all tiles from the original image
wsi.run(pipeline, distributed=True, client=client) # distributed=F, client=None

# Check how many tiles were generated from the image
print(f"Total number of tiles extracted: {len(wsi.tiles)}")

# Write the processed image file to data directory
wsi.write("./data/CMU-1-preprocessed.h5path")



dataset = TileDataset("./data/CMU-1-preprocessed.h5path")
dataloader = DataLoader(dataset, batch_size = 16, num_workers = 4)
