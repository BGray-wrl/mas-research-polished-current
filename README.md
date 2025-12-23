mas-research-polished-current

This is a working pipeline, but not a very polished one.

Install the required packages (requirements.txt).

To run the MAS, first create a .env file with ANTHROPIC_API_KEY and OPENAI_API_KEY.

Next, run the main runner from root with one of the provided configs. Example:
    `uv run src/main_runner.py --config configs/dummy.yaml`

I strongly encourage caution and reading the other configs before doing anything else. You will need a decrypted browse_comp_test.csv file in data/ to run other configs as well.

You can print metrics by changing the manual 'filepath' var in print_metrics.py to fit with the dummy output, the running
    `uv run src/print_metrics.py`          
and may change `test_local_filepath = "results/dummy-singleagent/current.json"` to whatever your testing output file is.          

Finally you can assemble metrics into a datafile by replacing 'analysis_metrics_filepaths.json' files with only valid paths and running
    `uv run src/assemble_run_metrics.py --results configs/analysis_metrics_filepaths.json --output FILENAME`
Metrics summary written to FILENAME

Manually change the paths in multirun_analysis.py (after name == __main__) then run
    `uv run src/mas_research/multirun_analysis.py`
To make the dataset *.csv file (and update is definitely todo, this is a wip)

And if that all works, manually changing the dataset path and
    `uv run experiments/experiments_and_visuals.py` 
Should work as well, generating visuals. It will also work with my example datafile data/dataset.csv


Note that this is all an early version, I intend to update the process soon and make it more robust.


## Notes/TODO
- `research_lead_agent.md` has 3 instances of the magic number `20`, one on line 84 and two on line 86, that I manually changed to `10` for the final round of browsecomp_plus runs. This has no impact on the paper as-is, but I would encourage making these changes if running evals on bowsecomp or browsecomp-plus.
- similarly, `researcher_agent_script.py` has `max_turns = 50` at line 83 for the agent workflow, but I set this to 25 for bc and bcp. This has no impact on the paper as is