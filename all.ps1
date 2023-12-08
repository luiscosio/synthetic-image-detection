# PowerShell script to run Python detection script with all models and datasets

# Define the path to your Python script
$pythonScript = "detection.py"

# Define the list of detector IDs
# $detectorIds = @("CNNDetector_p0.1_crop", "CNNDetector_p0.1", "CNNDetector_p0.5_crop", "CNNDetector_p0.5", "EnsembleDetector", "EnsembleDetector_crop", "CLIPDetector_crop", "DIRE")
$detectorIds = @("CNNDetector_p0.1_crop", "CNNDetector_p0.1", "CNNDetector_p0.5_crop", "CNNDetector_p0.5", "EnsembleDetector", "EnsembleDetector_crop", "CLIPDetector_crop")

# Define the list of dataset IDs
# $datasetIds = @("fast", "MSCOCO2014_val2014", "MSCOCO2014_valsubset", "MSCOCO2014_filtered_val", "SDR", "StableDiffusion2", "StableDiffusion2_ts20", "StableDiffusion2_ts80", "LDM", "Midjourney", "BigGAN", "StyleGAN2", "StyleGAN2r", "StyleGAN2f", "VQGAN", "Craiyon", "DALLE2", "DALLE3")
$datasetIds = @("MSCOCO2014_filtered_val", "Midjourney", "VQGAN", "DALLE2")

# Loop through each detector and dataset combination
foreach ($detectorId in $detectorIds) {
    foreach ($datasetId in $datasetIds) {
        # Construct the Python command
        $pythonCommand = "python `"$pythonScript`" --detector $detectorId --dataset $datasetId -bs 50 --verbose"
        
        # Optionally, add other parameters to $pythonCommand here

        # Run the Python script
        Invoke-Expression $pythonCommand
    }
}

# End of script
