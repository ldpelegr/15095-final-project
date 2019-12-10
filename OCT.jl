
using CSV, DataFrames #, Gurobi
#gurobi_env = Gurobi.Env()
ad = CSV.read("ML_project_data/ad.csv", header=false, categorical=true)
aps = CSV.read("ML_project_data/aps.data", header=false, categorical=true)
census = CSV.read("ML_project_data/census.csv", categorical=true)
covertype = CSV.read("ML_project_data/covertype.data", header=false, categorical=true)
heart = CSV.read("ML_project_data/heart_disease.data", header=false, categorical=true, missingstring="?")

for df in [ad, aps, census, covertype, heart]
    for i in 1:size(df,2)
        if typeof(df[:,i]) == Array{Union{Missing, String},1}
            df[:,i] = CategoricalArray(df[:,i])
        elseif typeof(df[:,i]) == Array{String,1}
            df[:,i] = CategoricalArray(df[:,i])
        end
    end
end

let i = 0
	global counter() = (i += 1)
end;

dfnames = ["ad", "aps", "census", "covertype", "heart"]
for df in [ad, aps, census, covertype, heart]
    i=counter()
    println(i)
    
    # Split data into features and outcomes
    X = df[:,1:size(df,2)-1]
    y = df[:,size(df,2)]

    # Split into training/validation/testing
    (train_X, train_y), (test_X, test_y) = IAI.split_data(:classification, X, y,
                                                          seed=2, train_proportion = 0.7);

    # define feature selection classifier
    FSgrid = IAI.GridSearch(
        IAI.OptimalFeatureSelectionClassifier(
            random_seed=2,
            ###  relaxation=false, 									### look into gurobi on the server for later
            ### solver=GurobiSolver(OutputFlag=0, gurobi_env)
        ),
        sparsity=1:min(20, size(X,2)),
    )

    # train feature selection model
    time_sparse = @elapsed IAI.fit!(FSgrid, train_X, train_y, validation_criterion=:auc);
    println("Time to fit sparse feature selection for ",  dfnames[i], ": ", time_sparse)


    ###############################################################################################################
    ## This is the block of code that pulls the list of features from the feature selection model (called "sparse")
    f = IAI.get_prediction_weights(FSgrid)
    sparse = append!(collect(keys(f[1])), collect(keys(f[2])))
    println("Features selected with optimal sparsity for ",  dfnames[i], ": ", sparse)
    ##
    ###############################################################################################################

    # redefine X to only include the features we've selected
    train_X_sparse = train_X[sparse]
    test_X_sparse = test_X[sparse]

    # fit an optimal tree classifier based on selected features
    lnr = IAI.OptimalTreeClassifier(random_seed=1)
    lnr.criterion = :entropy;
    grid = IAI.GridSearch(lnr,
        max_depth=4:10,
        minbucket=5
    )

    time_to_run = @elapsed IAI.fit!(grid, train_X_sparse, train_y, validation_criterion=:auc)
    println("Time to run OCT with sparse feature selection for ",  dfnames[i], ": ", time_to_run)

    println("Sparse tree AUC for ",  dfnames[i], ": ", IAI.score(grid, test_X_sparse, test_y, criterion=:auc))

    # write the tree visualization to a file
    IAI.write_html(dfnames[i] * "SparseTree.html", IAI.get_learner(grid))
    
    
    # fit regular tree for comparison
    lnr2 = IAI.OptimalTreeClassifier(random_seed=1)
    lnr2.criterion = :entropy;
    grid2 = IAI.GridSearch(lnr2,
        max_depth=4:10,
        minbucket=5
    )
    time_to_run = @elapsed IAI.fit!(grid2, train_X, train_y, validation_criterion=:auc)
    println("Time to run full OCT for ",  dfnames[i], ": ", time_to_run)

    println("Full tree AUC for ",  dfnames[i], ": ", IAI.score(grid2, test_X, test_y, criterion=:auc))

    # write the tree visualization to a file
    IAI.write_html(dfnames[i] * "FullTree.html", IAI.get_learner(grid2))
    
end

