mas-research-polished-current

This is a working pipeline, but not a very polished one.

Install the required packages (requirements.txt).

To run the MAS, first create a .env file with ANTHROPIC_API_KEY and OPENAI_API_KEY.

Next, run the main runner from root with one of the provided configs. Example:
    `uv run src/main_runner.py --config configs/dummy.yaml`

I strongly encourage caution and reading the other configs before doing anything else. You will need a decrypted browse_comp_test.csv file in data/ to run other configs as well.

You can print metrics by changing the manual 'filepath' var in print_metrics.py to fit with the dummy output, the running
    `uv run src/print_metrics.py`                       

Finally you can assemble metrics into a datafile by replacing 'analysis_metrics_filepaths.json' files with only valid paths and running
    `uv run src/assemble_run_metrics.py --results configs/analysis_metrics_filepaths.json --output FILENAME`
Metrics summary written to FILENAME

And if that all works, out-of-the-box 
    `uv run experiments/experiments_and_visuals.py` 
Should work as well, generating visuals. It will also work with my example datafile data/dataset.csv


Note that this is all an early version, I intend to update the process soon and make it more robust.