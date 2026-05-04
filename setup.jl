using Pkg

packages = [
    "AbstractGPs", "ApproximateGPs", "Arpack", "Combinatorics", 
    "Dierckx", "Distributions", "DoubleFloats", "ITensorMPS", 
    "ITensors", "Interpolations", "JLD2", "JSON", "KernelFunctions", 
    "LaTeXStrings", "LogExpFunctions", "NLopt", "NPZ", "Optim", 
    "Plots", "ProgressMeter", "PyCall", "PyPlot", "SpecialFunctions", 
    "StatsBase", "Yao", "Zygote"
]

println("Installing required Julia packages...")
Pkg.add(packages)
println("Setup complete!")
