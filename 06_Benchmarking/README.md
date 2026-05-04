## Configurations for reproducibility

To reproduce the study in the paper, please pass the following arguments when running the Julia scripts in `scripts/` in the order as follows. 
The first argument takes the number of qubits $n$, and the second argument takes the module index $m$.

### Data generation:
```bash
julia data_generation.jl 50 2
```

### Benchmarking:
```bash
julia qgp_run_bench.jl 50 2
```


### Further Notes:

 * To accommodate storage constraints, the dataset provided in the `data/` directory has been downsampled by a factor of two compared to the data used for the figures in the manuscript. This reduction in resolution introduces no noticeable qualitative or quantitative differences in the results.
 * To reproduce the kernel correction benchmarking results from scratch, you will need to execute the script `scripts/qgp_run_bench.jl` three separate times.<br>
 For each run, modify the `psd_method` keyword argument passed to `qgp_fit_predict` function calls in the script to use one of the following symbols. This will generate the corresponding `.jld2` summary files in the `results/` directory:
   * Set to `:trim` → generates `n50_m2_comparison_10_trim.jld2`
   * Set to `:shift` → generates `n50_m2_comparison_10_shift.jld2`
   * Set to `:semicircle` → generates `n50_m2_comparison_10_semicircle.jld2`
