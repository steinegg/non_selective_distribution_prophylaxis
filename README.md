# non_selective_distribution_prophylaxis

The numerical analysis presented in the manuscript directly follows the analytical calculations. The file “functions.jl” contains all the necessary functions to calculate the desired quantities. In particular, it finds the stationary state of the ODEs through the Julia package DifferentialEquations.  Additionally, by leveraging the packages Optim and DiffEqParamEstim, one can infer a transmission rate that leads to a desired prevalence level. With these functionalities the entire analysis is done in the file “immunization.jl”. The file requires some additional standard Julia libraries. Various routines are parallelized and were run on four cores of a MacBook Pro (M1, 2020). The analysis was performed in Julia version 1.5.3. The file “immunization.jl” is separated into different sections that each correspond to one of the figures in the manuscript. The sections are titled accordingly and are the following:

Section 1 - Plot Panel 1 A/B: Calculates the matrix J_ij to evaluate the functions F_dir(k), F_indir (k) and f(k) for the considered parameter set.  

Section 2 - Plot Panel 1 C: Evaluates the function f(k) for different values of efficacy. It further evaluates the dependency of k* on the efficacy.    

Section 3 - Plot Panel 2 A: Calculates k* as a function of prevalence and efficacy. For each prevalence value a corresponding transmission rate is inferred. The section further calculates the critical efficacies epsilon_c and epsilon_r. 

Section 4 - Plot Panel 2 B: Evaluates the critical efficacy (epsilon_c) as a function of the heterogeneity of the contact pattern (coefficient of variation) for different prevalence levels. 

Section 5 - Analyze world data Panel 3 B: Collects first all the necessary data to perform the analysis presented in Panel 3 B. The values extracted from the literature for different countries and cities are directly written in the file “immunization.jl”. Data from UN AIDS are read from the file “final_data.csv”. Once the data is put together, we calculate epsilon_c and epsilon_r for all the countries and cities.  

Section 6 - Plot Panel 3 A: Calculates epsilon_c and epsilon_r as a function of prevalence. Additionally, we calculate the prevalence at which epsilon_c and epsilon_r coincide with the considered PrEP efficacy of 60%. 

Section 7 - Supp Case Covid: Calculate epsilon_c as a function of prevalence for different values of the contact heterogeneity. The values of the contact heterogeneity interpolate from the over dispersion found for SARS-Cov-2 to a Poisson distribution. 

Section 8 - Supp Plot Panel 2 A Scale Free: The same as Section 3 but for a scale free contact structure. 

Section 9 - Supp Plot Panel 2 B Scale Free: The same as Section 4 but for a scale free contact structure. 

Section 10 - Supp Plot Panel 1 A/B Theory vs. Numerics: Numerical verification of the theory through  finite differences. Repeats the analysis in Figs. 1 A/B in the main text.  

Section 11 - Supp Plot Panel 1 C Theory vs. Numerics: Numerical verification of the theory through  finite differences. Repeats the analysis in Figs. 1 C in the main text.

Section 12 - Supp Plot Panel 1 A/B Assortativity: Repeats the analysis in Figs. 1 A/B in the main text for different levels of assortativity.

Section 13 - Supp Plot Panel 1 C Assortativity: Repeats the analysis in Figs. 1 C in the main text in the presence of assortativity.

Section 14 - Supp Plot Panel 3 A/B Assortativity: Repeats the analysis in Figs. 3 A/B in the main text in the presence of assortativity.  

With the exception of Section 5 and 14, the results are plotted similarly as in the manuscript but in more rudimental form. From the output of the file “immunization.jl”, the plots in the manuscript were done in a separate file in the programming language Python. 
