using PrecompileTools

@setup_workload begin
    @compile_workload begin
        ## This depends on RNG Implementation which is probably not ideal...
        Logging.with_logger(Logging.NullLogger()) do
            AutoFlux.fit(
                        AutoFlux.ImageClassification.ImageClassifier(;
                                popsize=2, 
                                insize=(32,32,3,0),
                                outsize=10
                            ), 
                        zeros(Float32, 32,32,3,0), # Zero size so we at least don't compute and gradients and stuff
                        zeros(Int, 10, 0); 
                        stopcriterion = pop -> generation(pop) > 1)
        end
    end
    Random.seed!(rng_default, 1)
end