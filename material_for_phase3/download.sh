while IFS='$\n' read -r line; do
    dbfs cp dbfs:/FileStore/shared_uploads/ram.senth@berkeley.edu/model_plots/$line ./
done